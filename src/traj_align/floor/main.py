"""Floor-alignment orchestration and public workflow entry points."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from traj_align.common.artifacts import (
    _load_detection_points, _transform_host_points,
    write_aligned_map_points_csv, write_aligned_observations,
    write_aligned_ply_pointcloud, write_aligned_tum_trajectory,
)
from traj_align import layout
from traj_align.common.geometry import estimate_floor_alignment
from traj_align.atomic_json import write_json_atomic
from .logging import LOGGER, _floor_diagnostic_log, _format_normal, _report_progress
from .open3d import _refine_floor_alignment_with_open3d
from .types import FloorAlignedArtifacts, FloorAlignment, VlmFloorSelectionError
from traj_align.common.types import PlaneDetectionConfig
from traj_align.timing import StageTimingCallback, timed_stage
from .vlm_images import _refine_floor_with_vlm


def _floor_vlm_client(config: PlaneDetectionConfig):
    """Build the configured VLM backend for one floor step.

    Imported here rather than at module scope on purpose: ``traj_align.vlm.client``
    is cheap to import, but it is the door to ``torch``/``transformers``.
    """

    from traj_align.vlm.client import OpenAiRetryPolicy, vlm_client

    base_url = (config.vlm_base_url or "").strip() or None
    if config.vlm_backend == "served":
        LOGGER.info(
            "[floor-vlm] %s VLM service endpoint: %s",
            config.vlm_local_profile or config.vlm_model or "default",
            base_url or "<none>",
        )
    return vlm_client(
        config.vlm_backend,
        model=config.vlm_model,
        local_profile=config.vlm_local_profile,
        api_key_file=config.openai_api_key_file,
        base_url=base_url,
        served_model=config.vlm_served_model,
        image_transport=config.vlm_image_transport,
        retry=OpenAiRetryPolicy(
            attempts=config.vlm_retry_attempts,
            initial_seconds=config.vlm_retry_initial_seconds,
            max_seconds=config.vlm_retry_max_seconds,
        ),
    )


def _aligned_output(source: Path | None, output_dir: Path, marker: str) -> Path | None:
    """``output_dir/<stem>.<marker><suffix>``: aligned copies never land beside the inputs."""

    if source is None:
        return None
    return output_dir / f"{source.stem}.{marker}{source.suffix}"


def _log_alignment(stage: str, points: np.ndarray, alignment: "FloorAlignment") -> None:
    """Log the fitted-plane residual and transform invariants at a decision boundary."""

    floor_index = alignment.planes.index(alignment.floor)
    normal = np.asarray(alignment.floor.normal, dtype=float)
    residuals = points @ normal + alignment.floor.offset
    transformed = _transform_host_points(points, alignment)
    selected = np.abs(residuals) <= 0.05
    selected_z = transformed[selected, 2]
    LOGGER.info(
        "[floor-vlm] %s alignment: plane=%d method=%s normal=%s offset=%.5f rotation_xyzw=%s translation=%s residual_abs(p50=%.5f p95=%.5f max=%.5f) selected_z(mean=%.5f std=%.5f count=%d).",
        stage,
        floor_index,
        (alignment.selection or {}).get("method"),
        _format_normal(alignment.floor.normal),
        alignment.floor.offset,
        tuple(round(value, 6) for value in alignment.rotation_xyzw),
        tuple(round(value, 6) for value in alignment.translation),
        float(np.percentile(np.abs(residuals), 50)),
        float(np.percentile(np.abs(residuals), 95)),
        float(np.abs(residuals).max()),
        float(selected_z.mean()) if len(selected_z) else float("nan"),
        float(selected_z.std()) if len(selected_z) else float("nan"),
        len(selected_z),
    )


def align_floor_artifacts(
    output_dir: Path,
    *,
    observations: Path | None,
    keyframe_trajectory: Path | None,
    camera_trajectory: Path | None,
    map_points: Path | None,
    pointcloud: Path | None,
    dataset: Path | None = None,
    camera: dict[str, float] | None = None,
    config: PlaneDetectionConfig | None = None,
    progress: Callable[[str], None] | None = None,
    stage_timing_callback: StageTimingCallback | None = None,
    cancellation_requested: Callable[[], bool] | None = None,
) -> FloorAlignedArtifacts | None:
    """Detect the floor plane and write aligned copies of map artifacts.

    The inputs are left intact; aligned copies are written into ``output_dir``
    as ``<stem>.floor_aligned<suffix>`` next to ``floor_alignment.json``.
    """

    config = config or PlaneDetectionConfig()

    def raise_if_cancelled() -> None:
        if cancellation_requested is not None and cancellation_requested():
            raise RuntimeError("Floor alignment cancelled.")

    key = jax.random.PRNGKey(config.seed)
    config = config.without_seed() # prevent false uses...

    with _floor_diagnostic_log(output_dir) as diagnostic_log:
        raise_if_cancelled()
        _report_progress(progress, "Loading map points for floor detection.")
        LOGGER.info("[floor-vlm] Starting floor alignment. diagnostic_log=%s config=%s", diagnostic_log, asdict(config))
        LOGGER.info(
            "[floor-vlm] Inputs: observations=%s keyframes=%s cameras=%s map_points=%s pointcloud=%s dataset=%s camera=%s",
            observations, keyframe_trajectory, camera_trajectory, map_points, pointcloud, dataset, camera,
        )
        with timed_stage(stage_timing_callback, "Floor: heuristic plane detection"):
            points, source = _load_detection_points(pointcloud=pointcloud, map_points=map_points)
            raise_if_cancelled()
            if len(points) < 3:
                # The one degrade this function has, and the only reason it ever
                # returns `None`: every other failure below raises. Callers rely on
                # that being unambiguous -- `stages.floor.align_floor_one` reads a
                # `None` as "this map cannot be aligned" rather than "the floor
                # module failed", and the two must not be reported the same way.
                LOGGER.error(
                    "[floor-vlm] NO FLOOR ALIGNMENT: only %d point(s) available from %s, and at "
                    "least 3 are needed to fit a plane. No aligned artifacts will be written, so "
                    "every consumer of this build -- Explore's point cloud included -- falls back "
                    "to the mapper's own arbitrary frame.",
                    len(points), source,
                )
                return None
            _report_progress(progress, f"Detecting structural planes from {len(points):,} map points.")
            LOGGER.info(
                "[floor-vlm] Loaded %d points from %s. bounds_min=%s bounds_max=%s centroid=%s.",
                len(points), source, np.round(points.min(axis=0), 4), np.round(points.max(axis=0), 4), np.round(points.mean(axis=0), 4),
            )
            geometry_points = jnp.asarray(points)
            key, rng = jax.random.split(key)
            alignment = estimate_floor_alignment(rng, geometry_points, config)
            raise_if_cancelled()
            _log_alignment("geometry", points, alignment)
        if config.vlm_enabled:
            # One client for both questions, closed however this block exits.
            # Selection and verification are one workflow asking the same model
            # up to four times; with the local backend a client per question
            # would pay the weight load (~28 s for the 8B) each time, and a
            # client left open would hold the whole L40S past the step, blocking
            # the next GPU step in a one-GPU allocation.
            client = _floor_vlm_client(config)
            try:
                if observations and observations.is_file():
                    _report_progress(progress, "Preparing visual evidence for VLM floor selection.")
                    with timed_stage(stage_timing_callback, "Floor: plane VLM selection"):
                        raise_if_cancelled()
                        key, rng = jax.random.split(key)
                        alignment = _refine_floor_with_vlm(
                            rng,
                            points,
                            alignment,
                            observations=observations,
                            output_dir=output_dir,
                            dataset=dataset,
                            camera=camera,
                            config=config,
                            client=client,
                            progress=progress,
                        )
                        raise_if_cancelled()
                else:
                    # Same silent-arm-substitution as every branch inside
                    # `_refine_floor_with_vlm`, one level up: the VLM cannot be
                    # asked at all, so the reason has to be recorded here too.
                    reason = "No observation manifest was available to build contact sheets from."
                    if config.vlm_required:
                        raise VlmFloorSelectionError(
                            f"VLM floor selection was configured but did not happen: {reason} "
                            "Set PlaneDetectionConfig.vlm_required=false to accept the geometry "
                            "floor, or vlm_enabled=false to select the geometry arm outright."
                        )
                    LOGGER.warning("[floor-vlm] %s Skipping contact-sheet selection and verifying geometry alignment directly.", reason)
                    alignment = alignment.with_vlm_error(reason)

                _report_progress(progress, "Preparing Open3D for VLM floor selection.")
                with timed_stage(stage_timing_callback, "Floor: Open3D VLM verification"):
                    raise_if_cancelled()
                    key, rng = jax.random.split(key)
                    alignment = _refine_floor_alignment_with_open3d(
                        rng, points, alignment, output_dir=output_dir, config=config, client=client
                    )
                    raise_if_cancelled()
            finally:
                client.close()
        with timed_stage(stage_timing_callback, "Floor: geometric finalization"):
            raise_if_cancelled()
            # Symmetry check, deliberately downstream of every selection pathway
            # and upstream of centering: the VLM chooses a plane, not a side of
            # one, so a correct choice can still leave the map under its floor.
            alignment = alignment.upright_from_point_support(geometry_points, config)
            if bool(alignment.selection.upright_flip_applied):
                LOGGER.warning(
                    "[floor-vlm] Inverted map: %.1f%% of off-plane points were below the selected "
                    "floor (threshold %.1f%%), so the cloud was rotated 180 degrees about X. The "
                    "selected plane is unchanged -- only which side of it the map sits on.",
                    100.0 * float(alignment.selection.upright_below_ratio),
                    100.0 * config.upright_below_fraction,
                )
                _log_alignment("upright", points, alignment)
            raise_if_cancelled()
            alignment = alignment.align_longest_wall_and_center(geometry_points, config)
            raise_if_cancelled()
            _log_alignment("final", points, alignment)
        _report_progress(progress, "Final floor transform selected; saving aligned map artifacts.")
        stem = layout.FLOOR_ALIGNED_MARKER

    with timed_stage(stage_timing_callback, "Floor: artifact export"):
        raise_if_cancelled()
        aligned_observations = output_dir / layout.FLOOR_ALIGNED_OBSERVATIONS if observations else None
        if observations and observations.is_file():
            _report_progress(progress, f"Saving {aligned_observations.name} to disk.")
            write_aligned_observations(observations, aligned_observations, alignment)
            raise_if_cancelled()
            _report_progress(progress, f"Saved {aligned_observations.name}.")

        aligned_keyframe = _aligned_output(keyframe_trajectory, output_dir, stem)
        if keyframe_trajectory and keyframe_trajectory.is_file() and aligned_keyframe:
            _report_progress(progress, f"Saving {aligned_keyframe.name} to disk.")
            write_aligned_tum_trajectory(keyframe_trajectory, aligned_keyframe, alignment)
            raise_if_cancelled()

        aligned_camera = _aligned_output(camera_trajectory, output_dir, stem)
        if camera_trajectory and camera_trajectory.is_file() and aligned_camera:
            _report_progress(progress, f"Saving {aligned_camera.name} to disk.")
            write_aligned_tum_trajectory(camera_trajectory, aligned_camera, alignment)
            raise_if_cancelled()

        aligned_map_points = _aligned_output(map_points, output_dir, stem)
        if map_points and map_points.is_file() and aligned_map_points:
            _report_progress(progress, f"Saving {aligned_map_points.name} to disk.")
            write_aligned_map_points_csv(map_points, aligned_map_points, alignment)
            raise_if_cancelled()

        aligned_pointcloud = _aligned_output(pointcloud, output_dir, stem)
        if pointcloud and pointcloud.is_file() and aligned_pointcloud:
            _report_progress(progress, f"Saving {aligned_pointcloud.name} to disk.")
            write_aligned_ply_pointcloud(pointcloud, aligned_pointcloud, alignment)
            raise_if_cancelled()

        metadata = output_dir / layout.FLOOR_ALIGNMENT_METADATA
        payload = {
            "source": str(source) if source else None,
            "config": asdict(config),
            "alignment": alignment.to_dict(),
            "artifacts": {
                "diagnostic_log": str(diagnostic_log),
                "observations": str(aligned_observations) if aligned_observations else None,
                "keyframe_trajectory": str(aligned_keyframe) if aligned_keyframe else None,
                "camera_trajectory": str(aligned_camera) if aligned_camera else None,
                "map_points": str(aligned_map_points) if aligned_map_points else None,
                "pointcloud": str(aligned_pointcloud) if aligned_pointcloud else None,
            },
        }
        _report_progress(progress, f"Saving {metadata.name} to disk.")
        write_json_atomic(metadata, payload)
        raise_if_cancelled()
        LOGGER.info("[floor-vlm] Wrote floor-alignment metadata: %s", metadata)
        _report_progress(progress, "Saved all floor-alignment artifacts.")

    return FloorAlignedArtifacts(
        metadata=metadata,
        observations=aligned_observations if aligned_observations and aligned_observations.exists() else None,
        keyframe_trajectory=aligned_keyframe if aligned_keyframe and aligned_keyframe.exists() else None,
        camera_trajectory=aligned_camera if aligned_camera and aligned_camera.exists() else None,
        map_points=aligned_map_points if aligned_map_points and aligned_map_points.exists() else None,
        pointcloud=aligned_pointcloud if aligned_pointcloud and aligned_pointcloud.exists() else None,
        alignment=alignment,
    )
