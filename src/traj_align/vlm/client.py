"""One seam for "ask a VLM about these images", local or remote.

Vendored from RAGMAP (``ragmap.vlm.client``) with RAGMAP's GPU-service lease
layer and captioning paths removed. Three backends, all asked the same question
through :class:`VlmRequest` and answering with :class:`VlmResponse`:

* ``local`` -- :class:`LocalVlmClient` loads a bundled profile
  (``traj_align/vlm/profiles/*.yaml``) onto this process's GPU with
  ``transformers``, answers, and frees the weights on ``close``.
* ``served`` -- :class:`ServedVlmClient` POSTs to any OpenAI-compatible
  ``/v1/chat/completions`` endpoint: a vLLM server, a ``ragmap-vlm-service``,
  or a hosted API (OpenRouter, Together, ...). Optional Bearer key.
* ``openai`` -- :class:`OpenAiVlmClient`, the OpenAI Responses API.

Nothing here imports ``torch``, ``transformers`` or ``openai`` at module scope;
each is imported inside the client that needs it.
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, Sequence

LOGGER = logging.getLogger("traj_align.vlm.client")

#: Backends ``vlm_client`` accepts. ``PlaneDetectionConfig.vlm_backend`` is a
#: plain string (it lives in a flax ``struct.dataclass`` that Hydra instantiates
#: from YAML), so the validation has to happen here rather than in a type.
#: ``local`` loads weights in the calling process; ``served`` is any
#: OpenAI-compatible chat-completions endpoint; ``openai`` is the Responses API.
BACKENDS = ("local", "openai", "served")


class VlmUnavailableError(RuntimeError):
    """The backend cannot be asked at all -- no key, no such profile, no SDK.

    Distinct from a request that was made and failed. Call sites turn this into
    the same "the VLM did not choose" degrade path they already had for a missing
    API key, so its message is human-facing text that ends up in
    ``floor_alignment.json``'s ``vlm_error`` and on the dashboard.
    """


@dataclass(frozen=True, slots=True)
class VlmImage:
    """One image, with optional text placed immediately before it.

    ``caption`` exists because the level-verification call does not just attach
    three renders -- it labels each one ("Render from positive x axis.") so the
    model can tell which view it is looking at. Dropping that for the local
    backend would silently ask a different question than the remote one, which
    is exactly the comparison the migration is validated against.
    """

    path: Path
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class VlmRequest:
    prompt: str
    images: tuple[VlmImage, ...]
    #: Cap on the answer, not the question. 1000 is what the Responses calls
    #: used; the answer is a one-line JSON object either way -- unless
    #: ``enable_thinking`` is set, in which case the reasoning is spent from this
    #: same budget and the caller must raise it (see
    #: ``PlaneDetectionConfig.vlm_answer_max_tokens``).
    max_tokens: int = 1_000
    #: Let a hybrid-reasoning checkpoint open a ``<think>`` block before
    #: answering. Only Qwen3.5 templates read it; see
    #: ``captioning_api.generate_conversations``. Part of the *request* rather
    #: than of the client because one client answers both floor questions and a
    #: campaign may want reasoning on the selection and not on the verification.
    enable_thinking: bool = False
    #: Which of the two floor questions this is, for log lines only.
    purpose: str = "vlm request"

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("VlmRequest requires a prompt")
        if not self.images:
            raise ValueError("VlmRequest requires at least one image")
        if self.max_tokens < 1:
            raise ValueError("VlmRequest.max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class VlmResponse:
    """A backend's answer plus what it could tell us about how it got there.

    ``diagnostics`` is deliberately free-form and deliberately not required to
    be comparable across backends: the Responses API can report reasoning-token
    counts and an ``incomplete_details.reason``, a local ``generate`` call
    cannot, and pretending otherwise would mean inventing values. Callers already
    treat every key as optional -- ``vlm_images._empty_vlm_response_message``
    reports whichever fields are present.
    """

    text: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


class VlmClient(Protocol):
    """Ask about images, once or several times, then release the resources."""

    def unavailable_reason(self) -> str | None:
        """A human-facing reason this client cannot answer, or ``None``.

        Separate from ``complete`` so a caller can refuse *before* doing
        expensive preparation -- ``open3d``'s verification renders three Open3D
        views per attempt and has always checked availability first.
        """

    def complete(self, request: VlmRequest) -> VlmResponse:
        """Answer ``request``, or raise. Raises :class:`VlmUnavailableError` when
        ``unavailable_reason`` is set; any other exception is a real failure of a
        real attempt, which callers record."""

    def describe_failure(self, exc: BaseException) -> tuple[str, dict[str, Any]]:
        """Turn an exception from ``complete`` into a human reason and diagnostics.

        Belongs to the backend, not the caller. The floor stage used to format
        every failure with ``openai_failure_summary``, which reads status/code/body
        off the exception -- correct for an API error and actively misleading for a
        local one, where it would record a processor bug as "OpenAI API request
        failed with ValueError". That string is what ends up in
        ``floor_alignment.json``'s ``vlm_error`` and on the dashboard, so it has to
        name what actually went wrong.
        """

    def close(self) -> None:
        """Release GPU weights / HTTP clients. Idempotent."""


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------


class LocalVlmClient:
    """Answer from a bundled profile's weights on this process's GPU.

    The model is loaded on the *first* ``complete`` and held until ``close``.
    That matters: one floor step asks up to four questions (one plane selection
    plus up to three level verifications), and the 8B costs ~28 s to load, so
    loading per call would spend most of the step in ``from_pretrained``.
    ``close`` then has to actually free the weights, because the step's SLURM
    allocation holds a *whole* L40S and the next GPU step cannot start until this
    process releases it.
    """

    def __init__(self, profile: str, *, cpu_threads: int = 2) -> None:
        self._profile = profile
        self._cpu_threads = cpu_threads
        self._model: Any = None
        self._processor: Any = None
        self._model_cfg: dict[str, Any] | None = None

    def _config(self) -> dict[str, Any]:
        if self._model_cfg is None:
            from .local import model_config

            self._model_cfg = model_config(self._profile)
        return self._model_cfg

    def unavailable_reason(self) -> str | None:
        try:
            self._config()
        except ValueError as exc:
            from .local import available_profiles

            return (
                f"Local VLM profile {self._profile!r} is not available ({exc}). "
                f"Bundled profiles: {', '.join(available_profiles())}."
            )
        try:
            # Cheap after the first call, and the answer has to be known *before*
            # the caller does expensive preparation: `open3d`'s verification
            # renders three Open3D views of the whole map per attempt.
            import torch  # noqa: F401 - name the extra, not a later NameError
            import transformers  # noqa: F401
        except ImportError as exc:
            return (
                f"The local VLM backend requires torch and transformers ({exc}); "
                "install them with `pip install 'traj-align[local]'`."
            )
        return None

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        reason = self.unavailable_reason()
        if reason:
            raise VlmUnavailableError(reason)
        from .local import configure_runtime, load_model_and_processor

        model_cfg = self._config()
        configure_runtime(self._cpu_threads)
        started = time.monotonic()
        processor_kwargs = dict(model_cfg.get("processor_kwargs", {}))
        self._model, self._processor = load_model_and_processor(
            model_cfg, self._cpu_threads, processor_kwargs
        )
        LOGGER.info(
            "[vlm-local] Loaded %s (revision %s) in %.1fs for profile %s.",
            model_cfg["id"], model_cfg["revision"], time.monotonic() - started, self._profile,
        )
        return self._model, self._processor

    def complete(self, request: VlmRequest) -> VlmResponse:
        # `_ensure_loaded` raises `VlmUnavailableError` before either import can
        # fail, so a missing extra is reported as "cannot be asked" rather than as
        # a failed attempt.
        model, processor = self._ensure_loaded()

        import torch

        from .local import generate_multi_image

        model_cfg = self._config()
        paths = tuple(image.path for image in request.images)
        captions = tuple(image.caption for image in request.images)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                text = generate_multi_image(
                    model, processor, paths, request.prompt, request.max_tokens,
                    image_captions=captions,
                    enable_thinking=request.enable_thinking,
                )
        except torch.cuda.OutOfMemoryError as exc:
            # Captioning halves its batch and retries. There is no equivalent
            # here: the images are the *alternatives in one question*, so
            # dropping some would change the question rather than the load. Say
            # what the operator can actually do about it.
            gc.collect()
            torch.cuda.empty_cache()
            raise VlmUnavailableError(
                f"CUDA out of memory asking {model_cfg['id']} about "
                f"{len(paths)} images at once (profile {self._profile!r}). Select a smaller "
                "profile via PlaneDetectionConfig.vlm_local_profile, or lower "
                "PlaneDetectionConfig.vlm_candidate_planes to send fewer contact sheets."
            ) from exc
        return VlmResponse(
            text=text,
            diagnostics={
                "backend": "local",
                "status": "completed",
                "model": str(model_cfg["id"]),
                "revision": str(model_cfg["revision"]),
                "image_count": len(paths),
                "generation_seconds": round(time.monotonic() - started, 3),
            },
        )

    def describe_failure(self, exc: BaseException) -> tuple[str, dict[str, Any]]:
        model = (self._model_cfg or {}).get("id")
        return (
            f"Local VLM ({self._profile}) failed with {type(exc).__name__}: {exc}",
            {
                "backend": "local",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "profile": self._profile,
                "model": str(model) if model else None,
            },
        )

    def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._processor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - only reachable without the extra
            pass
        LOGGER.info("[vlm-local] Released %s weights.", self._profile)

    def __enter__(self) -> "LocalVlmClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# OpenAI backend: classification, backoff, and the one call site
# ---------------------------------------------------------------------------

# Statuses worth another attempt. 429 is the one this exists for; 408 and the
# 5xx family are transient by definition. Deliberately excludes 400/401/403/404,
# where retrying burns the whole backoff budget on a request that cannot succeed.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
# Transport-level failures carry no status code, so they are matched by name --
# which also avoids importing `openai`'s exception classes just to classify, and
# keeps this working when the SDK renames or re-parents them.
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
)


#: 429s that are NOT throttling. OpenAI reports an exhausted credit balance or
#: quota as `RateLimitError` with status 429, which is indistinguishable from
#: real rate limiting by status alone -- and retrying it is hopeless: no amount of
#: backoff adds credit. Observed the hard way: a 15-arm matrix lost 123 of 150
#: jobs to `credit_balance_exhausted`, each after burning the full retry budget,
#: and the 429 status made the whole thing look like a concurrency problem.
_TERMINAL_ERROR_CODES = frozenset(
    {
        "credit_balance_exhausted",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "account_deactivated",
    }
)


def _openai_error_code(exc: BaseException) -> str | None:
    """The server's machine-readable error code, wherever the SDK put it.

    Checked in three places because the SDK surfaces it inconsistently across
    versions: an attribute, a flat body, or nested under `body["error"]`.
    """

    candidates: list[Any] = [getattr(exc, "code", None)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        candidates.append(body.get("code"))
        error = body.get("error")
        if isinstance(error, dict):
            candidates.append(error.get("code"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _is_retryable_openai_error(exc: BaseException) -> bool:
    """Whether another attempt could plausibly succeed.

    Code first: a billing/quota 429 is terminal however transient its status
    looks, and treating it as retryable turns an instant, legible "you are out of
    credit" into the same message six backoffs later. Then status, then exception
    name -- `RateLimitError` matches either way, but a subclass carrying an
    unexpected status is judged on the status, the more specific signal.
    """

    if _openai_error_code(exc) in _TERMINAL_ERROR_CODES:
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS_CODES
    return type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES


def _openai_retry_after_seconds(exc: BaseException) -> float | None:
    """The server's own instruction, when it sent one.

    A 429's ``Retry-After`` is better information than any local guess, so it
    raises the floor of the computed wait. Read defensively: the header is
    absent on most errors, and `httpx` header access differs across versions.
    """

    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:  # noqa: BLE001 - a non-mapping headers object is not worth failing over.
        return None
    if raw is None:
        return None
    try:
        seconds = float(str(raw).strip())
    except ValueError:
        # The HTTP-date form. Not worth parsing: the exponential wait below is
        # already a reasonable answer, and OpenAI sends the numeric form.
        return None
    return seconds if seconds >= 0 else None


def openai_failure_summary(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    details = []
    if status is not None:
        details.append(f"status {status}")
    if code:
        details.append(f"code {code}")
    suffix = f" ({', '.join(details)})" if details else ""
    message = openai_error_message(exc)
    message_suffix = f": {message}" if message else ""
    return f"OpenAI API request failed with {type(exc).__name__}{suffix}{message_suffix}."


def openai_failure_diagnostics(exc: BaseException) -> dict[str, Any]:
    """Record safe API failure fields alongside normal Responses completion diagnostics."""

    return {
        "exception_type": type(exc).__name__,
        "status_code": getattr(exc, "status_code", None),
        "code": getattr(exc, "code", None),
        "request_id": getattr(exc, "request_id", None),
        "message": openai_error_message(exc),
    }


def openai_error_message(exc: BaseException) -> str | None:
    """Extract the server's validation message without serializing an entire exception body."""

    body = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"]
    message = getattr(exc, "message", None)
    return message if isinstance(message, str) else None


def _response_diagnostics(response: Any) -> dict[str, Any]:
    """Extract the small set of Responses API fields needed to debug no-text results."""

    error = getattr(response, "error", None)
    incomplete = getattr(response, "incomplete_details", None)
    usage = getattr(response, "usage", None)
    output = getattr(response, "output", ()) or ()
    return {
        "backend": "openai",
        "response_id": getattr(response, "id", None),
        "status": getattr(response, "status", None),
        "error_code": getattr(error, "code", None),
        "error_message": getattr(error, "message", None),
        "incomplete_reason": getattr(incomplete, "reason", None),
        "output_types": [getattr(item, "type", type(item).__name__) for item in output],
        "output_statuses": [getattr(item, "status", None) for item in output],
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "reasoning_tokens": getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", None),
    }


def openai_api_key(key_file: str | None) -> str | None:
    """Read the API key from ``$OPENAI_API_KEY``, else from ``key_file``.

    There is deliberately no cwd fallback: pass an explicit path.
    """

    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    if not key_file:
        LOGGER.warning(
            "[vlm-openai] No OPENAI_API_KEY in the environment and no "
            "PlaneDetectionConfig.openai_api_key_file configured."
        )
        return None
    secrets = Path(key_file).expanduser()
    if not secrets.is_file():
        LOGGER.warning(
            "[vlm-openai] OPENAI_API_KEY is unset and the configured key file does not exist: %s",
            secrets,
        )
        return None
    key = secrets.read_text(encoding="utf-8").strip()
    if not key:
        LOGGER.warning("[vlm-openai] The configured OpenAI key file is empty: %s", secrets)
    return key or None


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


@dataclass(frozen=True, slots=True)
class OpenAiRetryPolicy:
    """Total attempts, not retries: 1 means no retry at all."""

    attempts: int = 6
    initial_seconds: float = 2.0
    max_seconds: float = 60.0


class OpenAiVlmClient:
    """The previous floor-stage code path, unchanged in behaviour.

    ``base_url`` is the whole reason a future vLLM (or any OpenAI-compatible)
    host needs no code change: point it at that server and set an arbitrary key.
    The installed SDK already honours ``$OPENAI_BASE_URL`` with no code at all,
    so passing it here is an *explicit override* of that environment variable
    rather than the only way to redirect the client.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key_file: str | None = None,
        base_url: str | None = None,
        retry: OpenAiRetryPolicy | None = None,
    ) -> None:
        self._model = model
        self._api_key_file = api_key_file
        self._base_url = base_url
        self._retry = retry or OpenAiRetryPolicy()

    def unavailable_reason(self) -> str | None:
        if not openai_api_key(self._api_key_file):
            # Wording preserved verbatim: it is asserted on in tests, persisted
            # into `floor_alignment.json`'s `vlm_error`, and surfaced on the
            # dashboard as the reason an arm degraded.
            return "OpenAI API key was not configured."
        return None

    def complete(self, request: VlmRequest) -> VlmResponse:
        api_key = openai_api_key(self._api_key_file)
        if not api_key:
            raise VlmUnavailableError("OpenAI API key was not configured.")
        try:
            import openai  # noqa: F401 - probe before building a request
        except ImportError as exc:
            # An absent extra is "cannot be asked", not "was asked and failed".
            # Translated here so callers have one exception type to handle instead
            # of an ImportError branch that only one backend can ever raise.
            raise VlmUnavailableError(
                "OpenAI Python SDK is not installed "
                "(pip install 'traj-align[openai]')."
            ) from exc
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.prompt}]
        for image in request.images:
            if image.caption:
                content.append({"type": "input_text", "text": image.caption})
            content.append(
                {"type": "input_image", "image_url": image_data_url(image.path), "detail": "high"}
            )
        response = _create_openai_response(
            api_key, self._model, content,
            purpose=request.purpose, retry=self._retry, base_url=self._base_url,
            max_output_tokens=request.max_tokens,
        )
        return VlmResponse(
            text=str(getattr(response, "output_text", "") or "").strip(),
            diagnostics=_response_diagnostics(response),
        )

    def describe_failure(self, exc: BaseException) -> tuple[str, dict[str, Any]]:
        return openai_failure_summary(exc), openai_failure_diagnostics(exc)

    def close(self) -> None:
        """Nothing to release: the SDK client is created per call."""

    def __enter__(self) -> "OpenAiVlmClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _create_openai_response(
    api_key: str,
    model: str,
    content: list[dict[str, Any]],
    *,
    purpose: str,
    retry: OpenAiRetryPolicy,
    base_url: str | None = None,
    max_output_tokens: int = 1_000,
) -> Any:
    """Issue one Responses call, retrying the failures that are worth retrying.

    Shared by floor selection and level verification: the two requests differ
    only in ``content``, and a retry policy that lived in only one of them would
    leave the other to die on the first 429.

    ``max_retries=0`` on the client is deliberate. The OpenAI SDK retries twice
    by default with its own short backoff, which would silently compose with
    tenacity's -- making the effective attempt count and total wait a product of
    two policies and impossible to reason about from the configuration. One
    policy, in one place.

    Raises ``ImportError`` when the extra is not installed, and re-raises the
    final API exception once the budget is spent, so callers keep recording
    failures exactly as they did before.
    """

    from openai import OpenAI
    from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

    attempts = max(1, int(retry.attempts))
    max_wait = max(0.0, float(retry.max_seconds))
    exponential = wait_random_exponential(
        multiplier=max(0.0, float(retry.initial_seconds)), max=max_wait
    )

    def wait(state: Any) -> float:
        computed = exponential(state)
        exc = state.outcome.exception() if state.outcome else None
        after = _openai_retry_after_seconds(exc) if exc is not None else None
        if after is None:
            return computed
        # Respect the server, but never sleep past the configured ceiling: a
        # mistaken multi-hour Retry-After would otherwise stall a SLURM step
        # until its wall clock ran out.
        return min(max(computed, after), max_wait)

    def before_sleep(state: Any) -> None:
        exc = state.outcome.exception() if state.outcome else None
        LOGGER.warning(
            "[vlm-openai] %s attempt %d/%d failed (%s); retrying in %.1fs",
            purpose,
            state.attempt_number,
            attempts,
            openai_failure_summary(exc) if exc is not None else "unknown error",
            getattr(state, "upcoming_sleep", 0.0) or 0.0,
        )

    client_kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": 0}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    retryer = Retrying(
        retry=retry_if_exception(_is_retryable_openai_error),
        wait=wait,
        stop=stop_after_attempt(attempts),
        before_sleep=before_sleep,
        reraise=True,
    )
    return retryer(
        client.responses.create,
        model=model,
        input=[{"role": "user", "content": content}],
        # A small, deterministic visual classification. Do not let reasoning
        # consume the whole output budget before the model emits its JSON.
        reasoning={"effort": "none"},
        text={"format": {"type": "json_object"}},
        max_output_tokens=max_output_tokens,
    )


# ---------------------------------------------------------------------------
# Served backend: a resident GPU service on this node
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServiceRetryPolicy:
    """Total attempts, not retries: 1 means no retry at all.

    Much tighter than :class:`OpenAiRetryPolicy` on purpose. These retries cover
    a loopback connection to a service that may have just been relaunched into a
    new generation, not rate limiting by a paid API -- so seconds, not minutes.
    """

    attempts: int = 4
    initial_seconds: float = 0.5
    max_seconds: float = 8.0


#: HTTP statuses worth another attempt against a resident service. 400 is
#: excluded deliberately: a malformed request will be malformed next time too,
#: and retrying it just delays a legible error.
_RETRYABLE_SERVICE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class ServedVlmClient:
    """Ask an OpenAI-compatible chat-completions endpoint.

    A separate backend rather than a reconfigured :class:`OpenAiVlmClient`,
    because that one speaks the *Responses* API with parameters that have no
    chat-completions equivalent. ``profile`` is the ``model`` field sent in each
    request: the served model's name (``Qwen/Qwen3-VL-8B-Instruct`` on vLLM, a
    profile name on ``ragmap-vlm-service``, a provider slug on a hosted API).

    Images travel as base64 data URLs by default, which every host accepts.
    ``file`` sends ``file://`` URLs instead -- cheaper when the server shares
    this filesystem (vLLM needs ``--allowed-local-media-path`` for that).

    Uses ``urllib`` rather than an HTTP library: this is one JSON POST.
    """

    def __init__(
        self,
        *,
        base_url: str,
        profile: str,
        api_key: str | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        image_transport: str = "base64",
        retry: ServiceRetryPolicy | None = None,
        timeout_seconds: float = 1800.0,
        unavailable: str | None = None,
    ) -> None:
        if image_transport not in ("file", "base64"):
            raise ValueError(
                f"Unknown image transport {image_transport!r}; expected 'file' or 'base64'."
            )
        # A reason the *caller* already knows -- typically a lease that could not
        # be obtained. Carried rather than raised at construction, because this
        # module's contract is that construction does no I/O and cannot fail: the
        # floor stage builds its client, renders contact sheets, writes its audit
        # blob, and only then discovers the backend is unusable. That ordering is
        # what makes a degrade legible instead of losing the evidence.
        self._unavailable = (unavailable or "").strip() or None
        self._base_url = (base_url or "").rstrip("/")
        self._profile = profile
        self._api_key = api_key
        self._processor_kwargs = dict(processor_kwargs or {})
        self._image_transport = image_transport
        self._retry = retry or ServiceRetryPolicy()
        self._timeout = timeout_seconds

    # -- protocol ---------------------------------------------------------

    def unavailable_reason(self) -> str | None:
        if self._unavailable:
            return self._unavailable
        if not self._base_url:
            return (
                "The `served` VLM backend needs PlaneDetectionConfig.vlm_base_url "
                "to point at an OpenAI-compatible chat-completions endpoint."
            )
        # Deliberately *not* a network probe. A service that is up but still in
        # `from_pretrained` is a normal state the request path handles by waiting,
        # and failing here would turn a 28 s warm-up into a degraded arm. When the
        # lease integration lands, readiness is already established before the step
        # starts.
        return None

    def image_url(self, path: Path) -> str:
        if self._image_transport == "base64":
            return image_data_url(path)
        return f"file://{path.resolve()}"

    def build_messages(self, request: VlmRequest) -> list[dict[str, Any]]:
        """Prompt first, then interleaved (caption, image) pairs.

        The same order ``generate_multi_image`` and the Responses path both use.
        It is not cosmetic: the archived comparison was measured on prompt-first
        content, so reordering would ask a different question than the recorded
        agreement figures describe.
        """

        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        for image in request.images:
            if image.caption:
                content.append({"type": "text", "text": image.caption})
            content.append(
                {"type": "image_url", "image_url": {"url": self.image_url(image.path)}}
            )
        return [{"role": "user", "content": content}]

    def build_body(self, request: VlmRequest) -> dict[str, Any]:
        return self.body_for(
            self.build_messages(request),
            request.max_tokens,
            enable_thinking=request.enable_thinking,
        )

    def body_for(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        *,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._profile,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if enable_thinking:
            # vLLM's own spelling for per-request chat-template arguments, chosen
            # for the same reason `mm_processor_kwargs` below is: a vLLM swap
            # should need no translation. Sent only when true so that every body
            # this client has ever produced for a non-reasoning request is
            # byte-identical to before -- the service defaults the flag to false.
            body["chat_template_kwargs"] = {"enable_thinking": True}
        if self._processor_kwargs:
            # vLLM's own spelling for per-request multimodal processor arguments,
            # chosen over a private field so a vLLM swap needs no
            # translation. Captioning's `max_pixels` arrives this way, and it is
            # part of the caption cache key, so it must reach the server rather
            # than be silently dropped.
            body["mm_processor_kwargs"] = dict(self._processor_kwargs)
        return body

    def complete(self, request: VlmRequest) -> VlmResponse:
        reason = self.unavailable_reason()
        if reason:
            raise VlmUnavailableError(reason)
        payload = self._post("/v1/chat/completions", self.build_body(request), purpose=request.purpose)
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("The VLM service returned no choices")
        message = choices[0].get("message") or {}
        return VlmResponse(
            text=str(message.get("content") or "").strip(),
            diagnostics=self._diagnostics(payload),
        )

    def describe_failure(self, exc: BaseException) -> tuple[str, dict[str, Any]]:
        if isinstance(exc, ServiceHttpError):
            return (
                f"The VLM service returned HTTP {exc.status}: {exc.message}",
                {"backend": "served", "status": exc.status, "endpoint": self._base_url},
            )
        return (
            f"The VLM service request failed with {type(exc).__name__}: {exc}",
            {"backend": "served", "endpoint": self._base_url},
        )

    def close(self) -> None:
        """Nothing to release: each request opens and closes its own connection.

        Note what this does *not* do -- it does not stop the service. The service
        outlives any one caller by design; its lifetime belongs to its own request
        clock, and asking it once is what starts that clock rather than ending it.
        """

    def __enter__(self) -> "ServedVlmClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- transport --------------------------------------------------------

    @staticmethod
    def _diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
        usage = payload.get("usage") or {}
        extra = payload.get("ragmap") or {}
        return {
            "backend": "served",
            "model": payload.get("model"),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"),
            "group": extra.get("group"),
        }

    def _post(
        self, route: str, body: Mapping[str, Any], *, purpose: str,
    ) -> dict[str, Any]:
        import json
        import random
        import time
        import urllib.error
        import urllib.request

        url = f"{self._base_url}{route}"
        encoded = json.dumps(body).encode("utf-8")
        attempts = max(1, int(self._retry.attempts))
        last: BaseException | None = None
        for attempt in range(1, attempts + 1):
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            request = urllib.request.Request(url, data=encoded, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = _service_error_detail(exc)
                error = ServiceHttpError(exc.code, detail)
                if exc.code not in _RETRYABLE_SERVICE_STATUS:
                    raise error from exc
                last = error
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                # A connection refusal is the expected shape of "the service was
                # relaunched into a new generation between our request and now", so
                # it is retried rather than surfaced.
                last = exc
            if attempt < attempts:
                wait = min(
                    self._retry.max_seconds,
                    self._retry.initial_seconds * (2 ** (attempt - 1)),
                )
                wait *= 0.5 + random.random() / 2.0
                LOGGER.warning(
                    "[vlm-served] %s attempt %d/%d failed (%s); retrying in %.1fs",
                    purpose, attempt, attempts, last, wait,
                )
                time.sleep(wait)
        assert last is not None  # noqa: S101 - the loop always records a failure
        raise last


class ServiceHttpError(RuntimeError):
    """A non-2xx answer from the resident service."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


def _service_error_detail(exc: Any) -> str:
    """Pull the service's own error message out of its JSON body."""

    import json

    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - fall back to the status line
        return str(getattr(exc, "reason", "") or "no detail")
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, Mapping):
        return str(error.get("message") or error)
    return str(error or payload)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def vlm_client(
    backend: str,
    *,
    model: str,
    local_profile: str,
    api_key_file: str | None = None,
    base_url: str | None = None,
    retry: OpenAiRetryPolicy | None = None,
    cpu_threads: int = 2,
    served_model: str | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
    image_transport: str = "base64",
) -> VlmClient:
    """Build the configured backend. Never touches the network or the GPU.

    Construction is deliberately free of I/O so a caller can create the client
    early, prepare expensive evidence (contact sheets, Open3D renders), write its
    audit trail, and only then discover that the backend is unusable -- which is
    the order the floor stage has always failed in, and the reason its audit blob
    survives a degrade.

    ``served`` sends ``served_model`` (default: ``local_profile``) as the request's
    model name, and a Bearer key from ``api_key_file`` / ``$OPENAI_API_KEY`` when
    one is set -- a local vLLM without auth simply gets none.
    """

    if backend not in BACKENDS:
        raise ValueError(f"Unknown VLM backend {backend!r}; expected one of {BACKENDS}.")
    if backend == "local":
        return LocalVlmClient(local_profile, cpu_threads=cpu_threads)
    if backend == "served":
        # `base_url` carries a different meaning per backend and that is
        # deliberate: for `openai` it is an override of the SDK's own
        # `$OPENAI_BASE_URL` resolution, while here it is the only way to reach
        # the service, so an empty value is a configuration error rather than a
        # fall back to some default host.
        return ServedVlmClient(
            base_url=base_url or "",
            profile=served_model or local_profile,
            # Optional: a local vLLM has no auth, so no key is not worth a warning.
            api_key=(
                openai_api_key(api_key_file)
                if api_key_file or os.environ.get("OPENAI_API_KEY") else None
            ),
            processor_kwargs=processor_kwargs,
            image_transport=image_transport,
        )
    return OpenAiVlmClient(
        model=model, api_key_file=api_key_file, base_url=base_url, retry=retry,
    )


def vlm_provenance(
    backend: str, *, model: str, local_profile: str, served_model: str | None = None
) -> str:
    """What to record as "the model" for this backend.

    The arms of a campaign are only comparable against each other when the model
    that chose their floor is written down: local runs are internally consistent
    but not comparable to the pre-migration ``gpt-5.6`` numbers.

    ``served`` records the served model name (falling back to the profile), not
    the endpoint: the endpoint is an address, not a model.
    """

    if backend == "openai":
        return model
    if backend == "served":
        return served_model or local_profile
    return local_profile


__all__ = [
    "BACKENDS",
    "LocalVlmClient",
    "OpenAiRetryPolicy",
    "OpenAiVlmClient",
    "ServedVlmClient",
    "ServiceHttpError",
    "ServiceRetryPolicy",
    "VlmClient",
    "VlmImage",
    "VlmRequest",
    "VlmResponse",
    "VlmUnavailableError",
    "image_data_url",
    "openai_api_key",
    "openai_error_message",
    "openai_failure_diagnostics",
    "openai_failure_summary",
    "vlm_client",
    "vlm_provenance",
]
