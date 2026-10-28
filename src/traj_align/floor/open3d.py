"""Independent Open3D-rendered verification for VLM floor choices."""
from __future__ import annotations
import os
import threading
from pathlib import Path
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from traj_align.utils.jax_batched_map import DEFAULT_BATCH_SIZE as _JAX_BATCH_SIZE, map_batches
from traj_align.atomic_json import write_json_atomic
from traj_align.common.artifacts import _transform_host_points
from traj_align.common.geometry import _plane_inlier_mask_jax
from .logging import LOGGER, _report_progress
from .types import FloorAlignment, FloorVerificationError, _VlmAlignmentVerification
from traj_align.common.types import PlaneDetectionConfig
from .vlm_images import _empty_vlm_response_message, _parse_json_object, _vlm_model_label
from traj_align.vlm.client import (
    VlmClient,
    VlmImage,
    VlmRequest,
    VlmUnavailableError,
)

_OPEN3D_CAPTURE_SLOT = threading.BoundedSemaphore(value=1)


def _refine_floor_alignment_with_open3d(
    key: jax.Array,
    points: np.ndarray,
    alignment: FloorAlignment,
    *,
    output_dir: Path,
    config: PlaneDetectionConfig,
    client: VlmClient,
    progress: Callable[[str], None] | None = None,
) -> FloorAlignment:
    """Verify a VLM-selected floor from three Open3D renders, at most three times.

    The visual model only decides whether the current frame is level or which
    detected plane is actually the floor. The rotation itself always comes from
    the selected plane's fitted normal, so a visual judgement cannot introduce
    an arbitrary, approximate transform.
    """

    # Selection and visual verification form one VLM workflow.  A geometry
    # fallback has no VLM choice to verify, and a VLM-selected floor is checked
    # here before it may be emitted -- when `open3d_verification_enabled` says so,
    # which by default it does not.
    #
    # **That default is off, and it is a retreat rather than a simplification.**
    # The loop's accept condition is `selected_plane_index == current_plane_index`
    # (see below): the model is shown a plane and passes it by naming the same one
    # back. That makes convergence a property of how readily a model revises its
    # answer, not of how level the floor is -- and it is measured. Over the
    # 8-scene Replica set on 2026-08-09, `qwen3_vl_8b` converged 8/8 (usually at
    # attempt 1, repeating itself while still reporting `floor_is_level: false`,
    # so it "passed" a floor it had just called crooked), while `qwen3_vl_32b`
    # converged 1/8 -- the other seven proposed a fresh plane every turn (5 -> 10
    # -> 0 on office1) and failed the build on `max_attempts_exhausted`.
    #
    # A check that fails the more capable model 7 times out of 8, for reasons
    # unrelated to the geometry, is not a check anyone can leave switched on. What
    # it needs is a termination condition that is about the floor -- accept when
    # the plane's normal is within tolerance of the verified up-axis, treat a
    # proposal cycle as a tie broken by `structural_score` -- rather than about
    # agreement with the previous turn. Until then this is opt-in.
    #
    # Turning it back on is one field, and it is worth doing on any arm where the
    # selection is in doubt:
    #     PlaneDetectionConfig.open3d_verification_enabled=true
    if (
        not config.open3d_verification_enabled
        or not config.vlm_enabled
        or config.vlm_refinement_max_attempts <= 0
    ):
        return alignment
    if int(alignment.selection.method) not in {1, 2}:
        return alignment
    # Checked before rendering, not at the first call: each attempt renders three
    # Open3D views of the whole map, and there is no point paying for that when
    # the backend cannot be asked at all.
    unavailable = client.unavailable_reason()
    if unavailable:
        raise FloorVerificationError(
            f"A VLM-selected floor requires Open3D verification, but the VLM is "
            f"unavailable: {unavailable}"
        )

    artifacts_dir = output_dir / "floor_plane_vlm" / "refinement"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    audit_path = artifacts_dir / "verification.json"
    attempts: list[dict[str, Any]] = []
    initial_plane_index = int(alignment.floor_index)
    current = alignment

    for attempt_number in range(1, config.vlm_refinement_max_attempts + 1):
        current_plane_index = int(current.floor_index)
        _report_progress(
            progress,
            f"VLM verification turn {attempt_number}/{config.vlm_refinement_max_attempts}: rendering Open3D views for plane {current_plane_index}.",
        )
        LOGGER.info(
            "[floor-vlm] Open3D verification pass %d/%d for plane %d.",
            attempt_number,
            config.vlm_refinement_max_attempts,
            current_plane_index,
        )
        try:
            key, rng = jax.random.split(key)
            screenshots = _capture_open3d_axis_views(
                rng,
                points,
                current,
                destination=artifacts_dir,
                attempt_number=attempt_number,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001 - retain diagnostics before failing the map build.
            error = f"Could not capture Open3D verification renders: {type(exc).__name__}: {exc}"
            LOGGER.warning("[floor-vlm] %s", error)
            attempts.append({"attempt": attempt_number, "plane_index": current_plane_index, "error": error})
            _write_open3d_verification_audit(
                audit_path, initial_plane_index, attempts, current, config, status="unavailable"
            )
            raise FloorVerificationError(
                f"VLM-selected floor could not be visually verified. Inspect {audit_path}. {error}"
            ) from exc

        _report_progress(
            progress,
            f"VLM verification turn {attempt_number}/{config.vlm_refinement_max_attempts}: asking {_vlm_model_label(config)} to validate plane {current_plane_index}.",
        )
        verification = _ask_vlm_to_verify_alignment(current, screenshots, config, client)
        attempt = {
            "attempt": attempt_number,
            "plane_index": current_plane_index,
            "screenshots": {name: str(path) for name, path in screenshots.items()},
            "accepted": verification.accepted,
            "selected_plane_index": verification.plane_index,
            "parsed": verification.parsed,
            "raw_response": verification.response,
            "error": verification.error,
            "response_diagnostics": verification.response_diagnostics,
        }
        attempts.append(attempt)
        _write_open3d_verification_audit(audit_path, initial_plane_index, attempts, current, config, status="in_progress")

        if verification.error:
            _report_progress(progress, f"VLM verification turn {attempt_number}/{config.vlm_refinement_max_attempts} returned no usable decision; retrying.")
            LOGGER.warning("[floor-vlm] Open3D verification pass %d failed: %s", attempt_number, verification.error)
            # A completed Responses API request can still contain no text—for
            # example, when its token budget was spent on reasoning.  That is
            # transient and must consume a normal verification retry rather
            # than being reported as an Open3D availability failure.
            continue
        if verification.accepted:
            _report_progress(progress, f"VLM verification turn {attempt_number}/{config.vlm_refinement_max_attempts}: plane {current_plane_index} accepted.")
            LOGGER.info("[floor-vlm] Open3D verification accepted plane %d as level.", current_plane_index)
            _write_open3d_verification_audit(audit_path, initial_plane_index, attempts, current, config, status="accepted")
            return current.with_open3d_verification("accepted", len(attempts))

        plane_index = verification.plane_index
        if plane_index == current_plane_index:
            _report_progress(progress, f"VLM verification turn {attempt_number}/{config.vlm_refinement_max_attempts}: plane {current_plane_index} reselected and accepted.")
            # The selected plane is rotated to +Z from its fitted normal before
            # every render. A verifier that reselects it has accepted the only
            # semantic decision it can make: that this is the walkable floor.
            # Models can otherwise mistake the raw candidate normal in the
            # prompt for the displayed frame and report a nonexistent tilt.
            LOGGER.info(
                "[floor-vlm] Open3D verification reselected current plane %d as the walkable floor; accepting its exact +Z alignment.",
                current_plane_index,
            )
            _write_open3d_verification_audit(audit_path, initial_plane_index, attempts, current, config, status="accepted")
            return current.with_open3d_verification("accepted", len(attempts))
        if plane_index is None:
            LOGGER.warning(
                "[floor-vlm] Open3D verification did not provide a new valid floor plane (received %s); retaining plane %d.",
                plane_index,
                current_plane_index,
            )
            break
        LOGGER.info(
            "[floor-vlm] Open3D verification selected plane %d; applying its exact fitted normal to +Z.",
            plane_index,
        )
        _report_progress(progress, f"VLM verification turn {attempt_number}/{config.vlm_refinement_max_attempts}: switching to plane {plane_index}.")
        current = current.with_selected_floor(plane_index, method=2)

    status = "max_attempts_exhausted" if len(attempts) == config.vlm_refinement_max_attempts else "rejected"
    LOGGER.error("[floor-vlm] Open3D verification finished without acceptance (%s).", status)
    _write_open3d_verification_audit(audit_path, initial_plane_index, attempts, current, config, status=status)
    raise FloorVerificationError(
        f"VLM-selected floor plane {initial_plane_index} was not accepted by Open3D verification "
        f"({status}). Inspect {audit_path}."
    )
def _capture_open3d_axis_views(
    key: jax.Array,
    points: np.ndarray,
    alignment: FloorAlignment,
    *,
    destination: Path,
    attempt_number: int,
    config: PlaneDetectionConfig,
) -> dict[str, Path]:
    """Render the aligned map from the positive X, Y, and Z directions with Open3D."""

    try:
        # Open3D uses EGL for its offscreen renderer. Force the surfaceless
        # backend before importing it so alignment never depends on an X server.
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific.
        raise RuntimeError("Open3D refinement requires the open3d package.") from exc

    # Sample down to the render budget FIRST: both the rigid-transform and
    # the floor-plane mask below are pointwise (order- and
    # neighbour-independent), so computing them on the sampled subset gives
    # exactly the same per-point values as computing them on the full cloud
    # and then indexing down -- at a fraction of the cost when the source
    # cloud has many more than 60,000 points.
    if len(points) > 60_000:
        selected_indices = np.asarray(jax.device_get(jnp.sort(
            jax.random.choice(key, len(points), (60_000,), replace=False)
        )))
        points = points[selected_indices]
    rendered_points = _transform_host_points(points, alignment)
    center = rendered_points.mean(axis=0)
    bounds = rendered_points.max(axis=0) - rendered_points.min(axis=0)
    map_size = max(float(np.linalg.norm(bounds)), 1.0)
    gizmo_scale = map_size * 0.5
    camera_distance = map_size * 1.5
    floor_mask = np.asarray(jax.device_get(map_batches(
        _plane_inlier_mask_jax,
        points,
        alignment.floor.normal,
        alignment.floor.offset,
        jnp.asarray(config.distance_tolerance),
        in_axes=(0, None, None, None),
        batch_size=_JAX_BATCH_SIZE,
    )))
    colors = np.tile(np.asarray((102, 177, 204), dtype=np.uint8), (len(rendered_points), 1))
    colors[floor_mask] = np.asarray((246, 184, 68), dtype=np.uint8)

    with _OPEN3D_CAPTURE_SLOT:
        return _capture_open3d_axis_views_locked(
            o3d,
            rendered_points,
            colors,
            center,
            map_size,
            gizmo_scale,
            camera_distance,
            destination,
            attempt_number,
            config,
        )


def _capture_open3d_axis_views_locked(
    o3d: Any,
    rendered_points: np.ndarray,
    colors: np.ndarray,
    center: np.ndarray,
    map_size: float,
    gizmo_scale: float,
    camera_distance: float,
    destination: Path,
    attempt_number: int,
    config: PlaneDetectionConfig,
) -> dict[str, Path]:
    """Capture a static scene without a browser, WebSocket server, or display."""

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        config.vlm_refinement_render_width, config.vlm_refinement_render_height
    )
    try:
        scene = renderer.scene
        scene.set_background((0.035, 0.04, 0.05, 1.0))
        points_geometry = o3d.geometry.PointCloud()
        points_geometry.points = o3d.utility.Vector3dVector(rendered_points)
        points_geometry.colors = o3d.utility.Vector3dVector(colors.astype(float) / 255.0)
        point_material = o3d.visualization.rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = max(map_size / 800.0, 2.0)
        scene.add_geometry("map_points", points_geometry, point_material)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=gizmo_scale, origin=center)
        axis_material = o3d.visualization.rendering.MaterialRecord()
        axis_material.shader = "defaultLit"
        scene.add_geometry("centroid_axes", axes, axis_material)
        grid_material = o3d.visualization.rendering.MaterialRecord()
        grid_material.shader = "unlitLine"
        grid_material.line_width = 1.0
        scene.add_geometry("floor_grid", _open3d_xy_grid(o3d, map_size), grid_material)

        views = {
            "x_axis": (np.asarray((1.0, 0.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "y_axis": (np.asarray((0.0, 1.0, 0.0)), np.asarray((0.0, 0.0, 1.0))),
            "z_axis": (np.asarray((0.0, 0.0, 1.0)), np.asarray((0.0, 1.0, 0.0))),
        }
        screenshots: dict[str, Path] = {}
        for view_name, (direction, up) in views.items():
            renderer.setup_camera(60.0, center, center + direction * camera_distance, up)
            image = np.asarray(renderer.render_to_image())
            path = destination / f"attempt_{attempt_number:02d}_{view_name}.png"
            Image.fromarray(image).save(path)
            screenshots[view_name] = path
        LOGGER.info("[floor-vlm] Wrote Open3D pass %d screenshots: %s", attempt_number, list(screenshots.values()))
        return screenshots
    finally:
        del renderer


def _open3d_xy_grid(o3d: Any, map_size: float) -> Any:
    """Create a compact XY reference grid at Z=0 for the verification renders."""

    extent = map_size * 0.7
    coordinates = np.linspace(-extent, extent, 21)
    points: list[tuple[float, float, float]] = []
    lines: list[tuple[int, int]] = []
    for coordinate in coordinates:
        start = len(points)
        points.extend(((-extent, coordinate, 0.0), (extent, coordinate, 0.0)))
        lines.append((start, start + 1))
        start = len(points)
        points.extend(((coordinate, -extent, 0.0), (coordinate, extent, 0.0)))
        lines.append((start, start + 1))
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([(0.28, 0.30, 0.34)] * len(lines))
    return grid


def _ask_vlm_to_verify_alignment(
    alignment: FloorAlignment,
    screenshots: dict[str, Path],
    config: PlaneDetectionConfig,
    client: VlmClient,
) -> _VlmAlignmentVerification:
    current_plane = alignment.planes.index(alignment.floor)
    candidate_lines = [
        f"Plane {index}: raw_mapper_normal={tuple(round(value, 4) for value in plane.normal)}, "
        f"support_ratio={plane.support_ratio:.4f}, extent={tuple(round(value, 3) for value in plane.extent)}"
        for index, plane in enumerate(alignment.planes)
    ]
    prompt = (
        "You are verifying whether an indoor 3D map's selected floor is horizontal. "
        "The attached renders view the already-aligned map from positive X, positive Y, and positive Z toward its centroid. "
        "The 3-axis gizmo is centered on the map and is half the map diagonal: X is red, Y is green, Z is blue. "
        "The grid is the current XY (horizontal) plane. Amber points belong to the currently selected floor candidate; blue points are all other map points. "
        f"The current floor candidate is plane {current_plane}. "
        "Every render was generated after rotating that candidate's fitted normal exactly onto displayed +Z; "
        "raw_mapper_normal is from the pre-alignment SLAM frame and must never be used to judge rendered level. "
        "If the amber plane is the true walkable floor and is horizontal, return only "
        '{"floor_is_level": true, "reason": "short reason", "confidence": 0.0}. '
        "Otherwise choose the plane index that is the true walkable floor. Do not request an arbitrary rotation; the system will use that plane's exact fitted normal. "
        "Return only valid JSON: "
        '{"floor_is_level": false, "plane_index": 0, "reason": "short reason", "confidence": 0.0}.\n\n'
        + "\n".join(candidate_lines)
    )
    request = VlmRequest(
        prompt=prompt,
        images=tuple(
            VlmImage(
                screenshots[view_name],
                caption=f"Render from positive {view_name.replace('_', ' ')}.",
            )
            for view_name in ("x_axis", "y_axis", "z_axis")
        ),
        max_tokens=config.vlm_answer_max_tokens,
        enable_thinking=config.vlm_thinking_enabled,
        purpose="level verification",
    )
    try:
        response = client.complete(request)
    except VlmUnavailableError as exc:
        # As in `vlm_images._ask_vlm_for_floor`: the backend was never asked.
        return _VlmAlignmentVerification(False, None, "", {}, str(exc))
    except Exception as exc:  # noqa: BLE001 - retain the failure in artifacts.
        # Post-backoff, as in `vlm_images._ask_vlm_for_floor`, and described by
        # the backend that failed rather than assumed to be an API error.
        error, diagnostics = client.describe_failure(exc)
        return _VlmAlignmentVerification(False, None, "", {}, error, diagnostics)
    diagnostics = response.diagnostics
    raw = response.text
    LOGGER.info("[floor-vlm] Raw Open3D verification response: %s", raw)
    if not raw:
        return _VlmAlignmentVerification(
            False,
            None,
            "",
            {},
            _empty_vlm_response_message(diagnostics),
            diagnostics,
        )
    parsed = _parse_json_object(raw)
    if not parsed:
        return _VlmAlignmentVerification(
            False,
            None,
            raw,
            {},
            "VLM response did not contain a JSON object.",
            diagnostics,
        )
    if parsed.get("floor_is_level") is True:
        return _VlmAlignmentVerification(True, None, raw, parsed, response_diagnostics=diagnostics)
    if parsed.get("floor_is_level") is not False:
        return _VlmAlignmentVerification(
            False,
            None,
            raw,
            parsed,
            "VLM response did not include a boolean floor_is_level.",
            diagnostics,
        )
    try:
        plane_index = int(parsed["plane_index"])
    except (KeyError, TypeError, ValueError):
        return _VlmAlignmentVerification(
            False,
            None,
            raw,
            parsed,
            "VLM response did not include an integer plane_index.",
            diagnostics,
        )
    if plane_index < 0 or plane_index >= len(alignment.planes):
        return _VlmAlignmentVerification(
            False,
            None,
            raw,
            parsed,
            f"VLM returned out-of-range plane_index {plane_index}.",
            diagnostics,
        )
    return _VlmAlignmentVerification(False, plane_index, raw, parsed, response_diagnostics=diagnostics)


def _write_open3d_verification_audit(
    destination: Path,
    initial_plane_index: int,
    attempts: list[dict[str, Any]],
    alignment: FloorAlignment,
    config: PlaneDetectionConfig,
    *,
    status: str,
) -> None:
    payload = {
        "backend": config.vlm_backend,
        "model": _vlm_model_label(config),
        "initial_plane_index": initial_plane_index,
        "current_plane_index": int(alignment.floor_index),
        "max_attempts": config.vlm_refinement_max_attempts,
        "status": status,
        "attempts": attempts,
    }
    write_json_atomic(destination, payload)
    LOGGER.info("[floor-vlm] Wrote Open3D verification audit: %s", destination)
