"""``traj-align``: floor-align one trajectory's artifacts from the command line."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from traj_align.common.types import PlaneDetectionConfig
from traj_align.vlm.client import BACKENDS


def _parse_intrinsics(value: str) -> dict[str, float]:
    """``fx,fy,cx,cy[,depth_scale]`` or a JSON object with those keys."""

    if value.lstrip().startswith("{"):
        return {key: float(item) for key, item in json.loads(value).items()}
    parts = [float(item) for item in value.split(",")]
    if len(parts) not in (4, 5):
        raise argparse.ArgumentTypeError("expected fx,fy,cx,cy[,depth_scale]")
    return dict(zip(("fx", "fy", "cx", "cy", "depth_scale"), parts))


def _coerce(field: dataclasses.Field, raw: str) -> Any:
    if raw.lower() in ("none", "null"):
        return None
    kind = str(field.type)
    if "bool" in kind:
        if raw.lower() not in ("true", "false", "1", "0"):
            raise SystemExit(f"--set {field.name}: expected a boolean, got {raw!r}")
        return raw.lower() in ("true", "1")
    if "int" in kind:
        return int(raw)
    if "float" in kind:
        return float(raw)
    return raw


def build_config(args: argparse.Namespace) -> PlaneDetectionConfig:
    fields = {field.name: field for field in dataclasses.fields(PlaneDetectionConfig)}
    overrides: dict[str, Any] = {
        "vlm_enabled": not args.no_vlm,
        "vlm_backend": args.backend,
    }
    for name, value in (
        ("vlm_local_profile", args.profile),
        ("vlm_model", args.model),
        ("vlm_served_model", args.served_model),
        ("vlm_base_url", args.base_url),
        ("openai_api_key_file", args.api_key_file),
        ("vlm_candidate_planes", args.candidates),
        ("open3d_verification_enabled", args.open3d_verify or None),
    ):
        if value is not None:
            overrides[name] = value
    for item in args.set:
        name, sep, raw = item.partition("=")
        if not sep or name not in fields:
            raise SystemExit(f"--set {item!r}: unknown PlaneDetectionConfig field {name!r}")
        overrides[name] = _coerce(fields[name], raw)
    return PlaneDetectionConfig(**overrides)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="traj-align",
        description="Detect the floor plane (RANSAC + VLM) and write floor-aligned copies of a "
        "trajectory's artifacts: z up, floor at z=0, longest wall on an axis, centered.",
    )
    io = parser.add_argument_group("inputs/outputs")
    io.add_argument("--output-dir", type=Path, required=True, help="Where aligned artifacts and floor_alignment.json go.")
    io.add_argument("--observations", type=Path, help="JSONL of {image, depth?, pose:{tx,ty,tz,qx,qy,qz,qw}} (camera-to-world). Needed for VLM selection.")
    io.add_argument("--camera-trajectory", type=Path, help="TUM trajectory (timestamp tx ty tz qx qy qz qw).")
    io.add_argument("--keyframe-trajectory", type=Path, help="TUM keyframe trajectory.")
    io.add_argument("--map-points", type=Path, help="CSV with x,y,z columns (used for plane detection if no --pointcloud).")
    io.add_argument("--pointcloud", type=Path, help="PLY point cloud (ascii or binary_little_endian); preferred for plane detection.")
    io.add_argument("--dataset", type=Path, help="Root that relative image/depth paths in --observations resolve against.")
    io.add_argument("--intrinsics", type=_parse_intrinsics, help="fx,fy,cx,cy[,depth_scale] or a JSON object.")

    vlm = parser.add_argument_group("VLM")
    vlm.add_argument("--no-vlm", action="store_true", help="Geometry only: skip VLM floor selection.")
    vlm.add_argument("--backend", choices=BACKENDS, default="local",
                     help="local: in-process HF weights; served: OpenAI-compatible chat endpoint (vLLM, hosted APIs); openai: Responses API.")
    vlm.add_argument("--profile", help="Bundled local profile (see traj_align/vlm/profiles). Default: qwen3_vl_32b.")
    vlm.add_argument("--served-model", help="Model name sent to a served endpoint (default: the profile name).")
    vlm.add_argument("--model", help="OpenAI model for --backend openai.")
    vlm.add_argument("--base-url", help="Endpoint for served (required) or openai (override).")
    vlm.add_argument("--api-key-file", help="API key file when $OPENAI_API_KEY is unset.")
    vlm.add_argument("--candidates", type=int, help="Candidate planes shown to the VLM (default 12).")
    vlm.add_argument("--open3d-verify", action="store_true", help="Also run the Open3D render-and-verify pass.")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override any PlaneDetectionConfig field (repeatable).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not (args.pointcloud or args.map_points):
        parser.error("one of --pointcloud or --map-points is required for plane detection")

    from traj_align.floor.main import align_floor_artifacts

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = align_floor_artifacts(
        args.output_dir,
        observations=args.observations,
        keyframe_trajectory=args.keyframe_trajectory,
        camera_trajectory=args.camera_trajectory,
        map_points=args.map_points,
        pointcloud=args.pointcloud,
        dataset=args.dataset,
        camera=args.intrinsics,
        config=build_config(args),
        progress=lambda message: print(f"[traj-align] {message}", flush=True),
    )
    if result is None:
        print("[traj-align] Not enough points to fit a floor plane; nothing written.")
        return 1
    print(f"[traj-align] Wrote {result.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
