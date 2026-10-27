"""In-process Hugging Face VLM loading and multi-image generation.

Extracted from RAGMAP's ``ragmap.vlm.captioning_api``: just what
:class:`traj_align.vlm.client.LocalVlmClient` needs (profile lookup, model
loading, one prompt over several images), without the caption cache and
batching. ``torch`` and ``transformers`` are imported inside the functions.
"""
from __future__ import annotations

import logging
import os
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import OmegaConf

LOGGER = logging.getLogger("traj_align.vlm.local")


def profile_root() -> Path:
    """The bundled model profiles (``traj_align/vlm/profiles/*.yaml``)."""

    return Path(str(files("traj_align.vlm").joinpath("profiles")))


def available_profiles() -> tuple[str, ...]:
    """Every bundled profile name, sorted -- the set ``model_config`` accepts."""

    return tuple(sorted(path.stem for path in profile_root().glob("*.yaml")))


def model_config(profile: str) -> dict[str, Any]:
    path = profile_root() / f"{profile}.yaml"
    if not path.is_file():
        raise ValueError(f"Unknown local VLM model profile: {profile}")
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid local VLM model profile: {profile}")
    return value


def configure_runtime(cpu_threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_processor(model_cfg: Mapping[str, Any], processor_kwargs: Mapping[str, Any]) -> Any:
    """Build just the processor for one set of ``processor_kwargs``.

    Split out of ``load_model_and_processor`` for the resident service, which
    holds *one* model and *several* processors: captioning bounds visual tokens
    with ``max_pixels: 802816`` while floor selection passes the profile's own
    kwargs, and ``max_pixels`` is part of the caption cache key, so the two cannot
    be quietly unified. A processor is tens of MB of CPU-side config against
    16-24 GiB of weights, so serving both from one loaded model is what makes a
    single shared service possible at all.
    """

    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        **dict(processor_kwargs),
    )


def max_memory_of(model_cfg: Mapping[str, Any], gpu_totals_gib: Sequence[float]) -> dict[int | str, str] | None:
    """The profile's GPU budget as a ``from_pretrained(max_memory=...)`` dict, or ``None``.

    Opt-in, per profile: a profile with neither key loads exactly as before (``device_map`` alone, so
    accelerate budgets every visible GPU's free memory *and the host RAM*). It exists for a model
    sharded across several GPUs in one process (``device_map: auto`` places whole decoder layers,
    pipeline style; ``qwen3_5_122b_a10b``). *gpu_totals_gib* is each visible GPU's total memory.

    - ``gpu_reserve_gib: R`` -- every visible GPU, minus R GiB kept free of weights for activations,
      the KV cache and the vision tokens. Hardware-agnostic: the same profile shards over the two H100s
      a service was given (2 x 79.1 GiB) or over four L40S (4 x 44.4 GiB).
    - ``max_memory_gib: {gpu index | "cpu": GiB}`` -- passed as is (tests: a GPU + CPU split on one card).

    Without a ``"cpu"`` entry nothing may land off-GPU: ``_check_placement`` refuses a load that did,
    so a model that does not fit fails at load instead of running from host memory at a crawl.
    """

    reserve, explicit = model_cfg.get("gpu_reserve_gib"), model_cfg.get("max_memory_gib")
    if reserve is None and explicit is None:
        return None
    if reserve is not None and explicit is not None:
        raise ValueError(f"{model_cfg['id']}: set gpu_reserve_gib or max_memory_gib, not both")
    if not model_cfg.get("device_map"):
        raise ValueError(f"{model_cfg['id']}: a GPU budget needs a device_map (e.g. device_map: auto)")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise ValueError(f"{model_cfg['id']}: max_memory_gib must be a mapping {{device: GiB}}")
        budget = {(key if key in ("cpu", "disk") else int(key)): float(gib) for key, gib in explicit.items()}
    else:
        budget = {index: total - float(reserve) for index, total in enumerate(gpu_totals_gib)}
    if not budget or any(gib <= 0 for gib in budget.values()):
        raise ValueError(f"{model_cfg['id']}: GPU budget must be positive, got {budget!r}")
    return {key: f"{gib:.2f}GiB" for key, gib in budget.items()}


def _check_compute_capability(model_cfg: Mapping[str, Any], gpus: int) -> None:
    """Refuse GPUs below the profile's ``min_compute_capability`` (opt-in; e.g. ``8.9``).

    For an FP8 checkpoint this is the difference between FP8 compute and a silent BF16 model:
    below 8.9 (A100) transformers' fp8 quantizer *dequantizes* the weights to BF16 with only a
    warning, which doubles the weight memory and changes the numerics of a sweep member.
    """

    floor = model_cfg.get("min_compute_capability")
    if floor is None:
        return
    import torch

    need = tuple(int(part) for part in str(floor).split("."))
    for index in range(gpus):
        have = torch.cuda.get_device_capability(index)
        if tuple(have) < need:
            raise RuntimeError(
                f"{model_cfg['id']} requires compute capability >= {floor} on every visible GPU, but GPU "
                f"{index} ({torch.cuda.get_device_name(index)}) is {have[0]}.{have[1]}"
            )


def _check_placement(model_cfg: Mapping[str, Any], model: Any, max_memory: Mapping[int | str, str]) -> None:
    """Log where the layers went; refuse CPU / disk placement the budget did not allow."""

    placement = Counter(str(device) for device in (getattr(model, "hf_device_map", None) or {}).values())
    LOGGER.info("[vlm] %s placement (modules per device): %s", model_cfg["id"], dict(placement))
    off_gpu = {device for device in placement if device in ("cpu", "disk")}
    if off_gpu - {key for key in max_memory if isinstance(key, str)}:
        raise RuntimeError(
            f"{model_cfg['id']} did not fit its GPU budget {dict(max_memory)}: modules placed on "
            f"{sorted(off_gpu)} ({dict(placement)})"
        )


def load_model_and_processor(model_cfg: Mapping[str, Any], cpu_threads: int, processor_kwargs: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText

    torch.set_num_threads(cpu_threads)
    required_gpus = int(model_cfg.get("required_gpus", 1))
    available_gpus = torch.cuda.device_count()
    if available_gpus < required_gpus:
        raise RuntimeError(
            f"{model_cfg['id']} requires {required_gpus} visible GPUs, but only "
            f"{available_gpus} are available"
        )
    required_memory_gib = float(model_cfg.get("minimum_total_gpu_memory_gib", 0))
    available_memory_gib = sum(
        torch.cuda.get_device_properties(index).total_memory
        for index in range(available_gpus)
    ) / 1024**3
    if available_memory_gib < required_memory_gib:
        raise RuntimeError(
            f"{model_cfg['id']} requires at least {required_memory_gib:g} GiB of "
            f"visible GPU memory, but only {available_memory_gib:.1f} GiB is available"
        )
    _check_compute_capability(model_cfg, available_gpus)
    processor = load_processor(model_cfg, processor_kwargs)
    quantization = dict(model_cfg.get("quantization", {}))
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
        "dtype": torch.bfloat16,
        "attn_implementation": model_cfg.get("attn_implementation", "sdpa"),
        "low_cpu_mem_usage": True,
    }
    if quantization:
        if quantization.get("method") != "bitsandbytes_4bit_nf4":
            raise ValueError(
                f"Unsupported quantization method: {quantization.get('method')!r}"
            )
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=bool(quantization.get("use_double_quant", True)),
            bnb_4bit_compute_dtype=getattr(
                torch, str(quantization.get("compute_dtype", "bfloat16")),
            ),
        )
    device_map = model_cfg.get("device_map")
    if device_map:
        model_kwargs["device_map"] = str(device_map)
    max_memory = max_memory_of(model_cfg, [
        torch.cuda.get_device_properties(index).total_memory / 1024**3 for index in range(available_gpus)
    ])
    if max_memory is not None:
        model_kwargs["max_memory"] = max_memory
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"],
        **model_kwargs,
    )
    if not device_map:
        model = model.to("cuda")
    if max_memory is not None:
        _check_placement(model_cfg, model, max_memory)
    model = model.eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, processor


def generate_conversations(
    model: Any,
    processor: Any,
    conversations: Sequence[Sequence[Mapping[str, Any]]],
    max_tokens: int,
    *,
    enable_thinking: bool = False,
) -> list[str]:
    """Greedy-decode one completion per conversation.

    ``enable_thinking`` reaches the chat template (Qwen3.5 opens a ``<think>``
    block unless it is passed as false); templates that do not reference it
    ignore it. See ``PlaneDetectionConfig.vlm_thinking_enabled``.
    """

    import torch

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", "left") != "left":
        # Decoder-only batched generation must pad on the LEFT: generation
        # continues from the last position of every row, and the slice below
        # assumes each prompt ends at `input_ids.shape[1]`. Qwen3-VL's tokenizer
        # ships `padding_side: right`, so a batch of prompts of unequal length
        # -- text-only requests (Embodied-RAG's area summaries, sent
        # concurrently) are the first caller to form one -- decoded garbage for
        # every shorter row ("user\nYou are", ""), measured 2026-09-28. A batch
        # of equal-length prompts carries no padding, so its output is
        # unchanged by this; only a padded batch changes, from wrong to right.
        tokenizer.padding_side = "left"
    template_kwargs: dict[str, Any] = {"enable_thinking": enable_thinking}
    inputs = processor.apply_chat_template(
        list(conversations), add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True,
        **template_kwargs,
    ).to("cuda", dtype=torch.bfloat16)
    generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_tokens, use_cache=True)
    return processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_multi_image(
    model: Any,
    processor: Any,
    paths: Sequence[Path],
    prompt: str,
    max_tokens: int,
    *,
    image_captions: Sequence[str | None] | None = None,
    enable_thinking: bool = False,
) -> str:
    """Answer *one* prompt about *all* of ``paths`` at once.

    Not a variant of ``generate_batch``, the opposite of it. ``generate_batch``
    builds N independent single-image conversations and returns N answers --
    exactly right for captioning, where the images have nothing to do with each
    other. Floor-plane selection needs the other shape: six contact sheets are
    the *alternatives in one question*, and the answer ("plane 3") only exists
    relative to the whole set. Feeding them through ``generate_batch`` would ask
    six separate questions and get six unrelated answers.

    ``image_captions``, when given, is text placed immediately *before* the
    corresponding image, which is how the level-verification question tells the
    model which of three renders it is looking at.

    The prompt comes first, then the images: the content order the OpenAI
    Responses calls use, so every backend is asked the same question.

    Greedy, like every other generation path here: the floor frame propagates
    into every downstream artifact, so the same map must select the same plane.
    """

    if not paths:
        raise ValueError("generate_multi_image requires at least one image")
    captions = list(image_captions) if image_captions is not None else [None] * len(paths)
    if len(captions) != len(paths):
        raise ValueError(
            f"generate_multi_image got {len(captions)} image_captions for {len(paths)} images"
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path, caption in zip(paths, captions, strict=True):
        if caption:
            content.append({"type": "text", "text": caption})
        content.append({"type": "image", "path": str(path)})
    conversation = [{"role": "user", "content": content}]
    decoded = generate_conversations(
        model, processor, [conversation], max_tokens, enable_thinking=enable_thinking
    )
    if len(decoded) != 1:
        raise RuntimeError(f"Expected one completion for a single conversation, got {len(decoded)}")
    return decoded[0]
