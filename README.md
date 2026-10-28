# traj-align

VLM-assisted floor detection and trajectory alignment, extracted from
RAGMAP (`ragmap.alignment.floor` + `ragmap.alignment.common`,
ragmap commit `4954f41`). Cross-trajectory scene stitching
(`ragmap.alignment.scene`) is intentionally **not** included.

Given one trajectory's map, it:

1. **Detects structural planes** with JAX RANSAC over a point cloud / map points.
2. **Asks a VLM which plane is the floor.** It renders one contact sheet per candidate
   plane (the plane's points projected into sampled camera frames) and asks for the floor in
   one multi-image question. Optional multi-round elimination and an optional
   Open3D render-and-verify pass are available.
3. **Finalizes the frame.** It applies an upright flip if the map sits under its floor,
   aligns the longest wall to an axis, and centers the map, giving z up with the floor at z=0.
4. **Writes aligned copies** of the observations, TUM trajectories, map-point CSV and PLY,
   plus `floor_alignment.json`, an audit of every plane, VLM prompt/answer and transform.

## VLM backends

| `--backend` | What it is | Needs |
|---|---|---|
| `local` (default) | Hugging Face weights loaded in-process from a bundled profile (`src/traj_align/vlm/profiles/*.yaml`) and freed afterwards | `.[local]`, a GPU: `qwen3_vl_32b` (NF4) runs on an L40S (verified, ~80 s load + ~12 s answer) but OOMs on a 46 GiB Quadro RTX 8000 with 12 sheets |
| `served` | Any OpenAI-compatible `/v1/chat/completions` endpoint, such as vLLM, `ragmap-vlm-service`, OpenRouter or Together | `--base-url`; optional Bearer key via `$OPENAI_API_KEY` / `--api-key-file`; `--served-model` names the model |
| `openai` | OpenAI Responses API | `.[openai]`, `$OPENAI_API_KEY` |

The default profile is `qwen3_vl_32b` with 12 candidate planes, the configuration RAGMAP
ships. `--no-vlm` gives the geometry-only result.

## Install

```bash
uv venv --python 3.12 && uv pip install -e '.[local,openai,open3d]'   # or just the extras you need
```

On Mila, `[local]` (torch) does not fit in the home quota: put the venv on scratch
(`uv venv $SCRATCH/venvs/traj-align && ln -s $SCRATCH/venvs/traj-align .venv`), or run
with an existing environment that already has the stack, e.g.
`PYTHONPATH=src ~/ragmap/.venv/bin/python -m traj_align.cli ...`.

## Usage

```bash
# local weights
traj-align --output-dir out/ --pointcloud map.ply --observations observations.jsonl \
  --camera-trajectory CameraTrajectory.txt --intrinsics 600,600,320,240,1000 -v

# a vLLM server (or any OpenAI-compatible host)
traj-align ... --backend served --base-url http://localhost:8000 \
  --served-model Qwen/Qwen3-VL-32B-Instruct

# OpenAI
OPENAI_API_KEY=... traj-align ... --backend openai --model gpt-5.6

# any PlaneDetectionConfig field
traj-align ... --set vlm_selection_rounds=3 --set vlm_prompt_style=survey
```

Python:

```python
from pathlib import Path
from traj_align import PlaneDetectionConfig, align_floor_artifacts

result = align_floor_artifacts(
    Path("out"), observations=Path("observations.jsonl"), keyframe_trajectory=None,
    camera_trajectory=Path("CameraTrajectory.txt"), map_points=None, pointcloud=Path("map.ply"),
    camera={"fx": 600, "fy": 600, "cx": 320, "cy": 240, "depth_scale": 1000},
    config=PlaneDetectionConfig(vlm_backend="served", vlm_base_url="http://localhost:8000"),
)
print(result.alignment.to_dict())
```

## Input formats

- **Point cloud / map points**: you need at least one for plane detection. A PLY can be
  `ascii` or `binary_little_endian` with `x,y,z` vertex properties. A CSV needs `x,y,z`
  columns.
- **Observations** (JSONL, required for VLM selection): one object per frame,
  `{"image": "rgb/000.png", "depth": "depth/000.png", "pose": {"tx","ty","tz","qx","qy","qz","qw"}}`.
  The pose is **camera-to-world** (`Twc`) in the map frame, and `depth` is optional. Relative
  paths resolve against `--dataset`, falling back to the JSONL's own directory.
- **Trajectories**: TUM format, `timestamp tx ty tz qx qy qz qw`.
- **Intrinsics**: `fx,fy,cx,cy[,depth_scale]`. Without them, `fx=fy=600` and the image
  center are assumed.

Outputs go to `--output-dir` as `<input stem>.floor_aligned<suffix>`,
`observations.floor_aligned.jsonl` and `floor_alignment.json`.

## Differences from RAGMAP

- RAGMAP's GPU-service lease layer is removed. `served` simply POSTs to `--base-url`.
  RAGMAP's caption cache and batching are also gone.
- `served` gains an optional Bearer key, `vlm_served_model`, and base64 image transport
  by default (`vlm_image_transport`; RAGMAP sent `file://` URLs to a same-node service).
- `PlaneDetectionConfig` defaults follow RAGMAP's shipped Hydra profile
  (`qwen3_vl_32b`, `vlm_candidate_planes=12`), but the backend defaults to `local`.
- Aligned copies are written into `output_dir`. RAGMAP wrote them beside the inputs.
- The plugin wrappers (`FloorAligner` protocol, the `passthrough` ablation arm, Hydra
  configs) are not included.

The code comments still describe RAGMAP campaigns and measurements. Those are kept as
the design history behind the defaults.
