# ruff: noqa: F401
"""Floor-plane recognition and map-frame alignment helpers.

Separates geometric estimation, VLM image selection, Open3D verification,
artifact transformation, and orchestration.
"""
from traj_align.common.artifacts import (
    transform_point, transform_pose_dict,
    _transform_ply_coordinates_in_place,
    write_aligned_map_points_csv, write_aligned_observations,
    write_aligned_ply_pointcloud, write_aligned_tum_trajectory,
)
from traj_align.common.geometry import (
    estimate_floor_alignment,
)
from .main import align_floor_artifacts
from .open3d import (
    _ask_vlm_to_verify_alignment,
    _capture_open3d_axis_views,
    _refine_floor_alignment_with_open3d,
)
from .types import (
    FloorAlignedArtifacts, FloorAlignment, FloorSelection, FloorVerificationError, _VlmAlignmentVerification,
)
from traj_align.common.types import PlaneSet
from .vlm_images import (
    _refine_floor_with_vlm,
    _score_projected_plane,
)

__all__ = [
    "FloorAlignedArtifacts", "FloorAlignment", "FloorSelection", "FloorVerificationError", "PlaneSet",
    "align_floor_artifacts",
    "estimate_floor_alignment", "transform_point", "transform_pose_dict",
    "write_aligned_map_points_csv", "write_aligned_observations",
    "write_aligned_ply_pointcloud", "write_aligned_tum_trajectory",
]
