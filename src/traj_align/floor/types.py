"""Shared floor-alignment data types."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from flax import struct
from jax import lax
from jax.scipy.spatial.transform import Rotation

from traj_align.common.types import (
    MapCentering,
    PlaneDetectionConfig,
    PlaneSet,
    PlaneSummary,
    Pose,
    WallAlignment,
)
from traj_align.utils.jax_batched_map import DEFAULT_BATCH_SIZE as _JAX_BATCH_SIZE, map_batches


_Z_AXIS = jnp.asarray((0.0, 0.0, 1.0))
#: 180 degrees about X: keeps the levelled floor at z=0 and swaps which side of
#: it the rest of the map sits on. See ``FloorAlignment.upright_from_point_support``.
_X_FLIP = jnp.diag(jnp.asarray((1.0, -1.0, -1.0)))
_JAX_EPSILON = 1e-9


@jax.jit
def _rotation_between_jax(source: jax.Array, target: jax.Array) -> jax.Array:
    """Return a matrix that maps ``source`` onto ``target``."""

    source = source / jnp.maximum(jnp.linalg.norm(source), _JAX_EPSILON)
    target = target / jnp.maximum(jnp.linalg.norm(target), _JAX_EPSILON)
    cross = jnp.cross(source, target)
    dot = jnp.clip(jnp.dot(source, target), -1.0, 1.0)
    cross_norm = jnp.linalg.norm(cross)
    identity = jnp.eye(3, dtype=source.dtype)

    def parallel(_: None) -> jax.Array:
        return identity

    def antiparallel(_: None) -> jax.Array:
        axis = jnp.where(
            jnp.abs(source[0]) > 0.9,
            jnp.array((0.0, 1.0, 0.0)),
            jnp.array((1.0, 0.0, 0.0)),
        )
        basis = jnp.cross(source, axis)
        basis = basis / jnp.maximum(jnp.linalg.norm(basis), _JAX_EPSILON)
        return 2.0 * jnp.outer(basis, basis) - identity

    def general(_: None) -> jax.Array:
        axis = cross / cross_norm
        return Rotation.from_rotvec(axis * jnp.arccos(dot)).as_matrix()

    return lax.cond(
        cross_norm <= _JAX_EPSILON,
        lambda _: lax.cond(dot >= 0.0, parallel, antiparallel, operand=None),
        general,
        operand=None,
    )


class FloorVerificationError(RuntimeError):
    """Raised when a VLM-selected floor cannot be visually verified.

    A VLM choice is only a proposal. Returning it after the independent Open3D
    check failed would mark a potentially invalid map as aligned.
    """


class VlmFloorSelectionError(RuntimeError):
    """Raised when a configured VLM never got to choose the floor plane.

    Distinct from :class:`FloorVerificationError`, which is about a choice the
    VLM *did* make. This one covers every path on which ``vlm_enabled`` was set
    and the geometry heuristic silently supplied the answer instead -- no API
    key, no SDK, a failed request, an unparseable reply, or no visual evidence
    to send. Those are all defensible fallbacks for a single map and indefensible
    for an ablation, so ``PlaneDetectionConfig.vlm_required`` decides which.
    """


@struct.dataclass
class Open3DVerification:
    status: jax.Array
    attempt_count: jax.Array


_METHOD_NAMES = ("geometry", "vlm", "vlm+open3d", "passthrough")
_VERIFICATION_NAMES = ("not_requested", "in_progress", "accepted", "unavailable", "rejected")


@struct.dataclass
class FloorSelection:
    """Numeric decision state. ``to_dict`` is only for JSON/audit boundaries."""

    method: jax.Array
    plane_index: jax.Array
    wall_alignment: WallAlignment
    centering: MapCentering
    open3d_verification: Open3DVerification
    #: Whether the selected plane's normal was flipped to put the map above its
    #: floor, and the share of off-plane points that were below it at the time.
    #: A flip is invisible in every other recorded field -- the plane index does
    #: not change and the rotation is just as valid a levelling either way -- so
    #: this is the only place a reader can tell an inverted map was corrected.
    upright_flip_applied: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(False)
    )
    upright_below_ratio: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0))
    #: Why the VLM did not choose this floor, when one was configured. Static
    #: (never a PyTree leaf): it is audit text, and ``None`` on every path where
    #: the VLM either was not asked for or answered. ``stages.floor
    #: .floor_alignment_status`` reads it back out of ``floor_alignment.json``
    #: to say so on the unit status and the dashboard event stream.
    vlm_error: str | None = struct.field(pytree_node=False, default=None)

    @classmethod
    def geometry(cls, plane_index: int | jax.Array) -> "FloorSelection":
        return cls(
            method=jnp.asarray(0, dtype=jnp.int32),
            plane_index=jnp.asarray(plane_index, dtype=jnp.int32),
            wall_alignment=WallAlignment(
                applied=jnp.asarray(False),
                plane_index=jnp.asarray(-1, dtype=jnp.int32),
                normal=jnp.zeros(3),
                inliers=jnp.asarray(0, dtype=jnp.int32),
                yaw_degrees=jnp.asarray(0.0),
                perpendicular_tolerance_degrees=jnp.asarray(0.0),
            ),
            centering=MapCentering(source_centroid=jnp.zeros(3), translation=jnp.zeros(3)),
            open3d_verification=Open3DVerification(
                status=jnp.asarray(0, dtype=jnp.int32),
                attempt_count=jnp.asarray(0, dtype=jnp.int32),
            ),
        )

    @classmethod
    def from_value(
        cls, selection: "FloorSelection | Mapping[str, Any] | None", plane_index: int | jax.Array
    ) -> "FloorSelection":
        """Normalize legacy selection metadata at the alignment boundary."""

        plane_index = jnp.asarray(plane_index, dtype=jnp.int32)
        if isinstance(selection, cls):
            return selection.replace(plane_index=plane_index)
        method = (selection or {}).get("method", "geometry")
        method_code = {"geometry": 0, "vlm": 1, "vlm+open3d": 2, "passthrough": 3}.get(method, 0)
        return cls.geometry(plane_index).replace(
            method=jnp.asarray(method_code, dtype=jnp.int32),
            vlm_error=(selection or {}).get("vlm_error"),
        )

    def with_selected_plane(self, plane_index: int | jax.Array, method: int) -> "FloorSelection":
        """Return this audit state for a newly selected floor plane."""

        return self.replace(
            method=jnp.asarray(method, dtype=jnp.int32),
            plane_index=jnp.asarray(plane_index, dtype=jnp.int32),
        )

    def with_wall_alignment(
        self, wall_alignment: WallAlignment, centering: MapCentering
    ) -> "FloorSelection":
        """Return this audit state with its final wall and centering decisions."""

        return self.replace(wall_alignment=wall_alignment, centering=centering)

    def with_upright_flip(self, applied: jax.Array, below_ratio: jax.Array) -> "FloorSelection":
        """Return this audit state with its floor-side (symmetry) decision."""

        return self.replace(upright_flip_applied=applied, upright_below_ratio=below_ratio)

    def with_vlm_error(self, error: str) -> "FloorSelection":
        """Return this audit state with the reason the VLM did not choose."""

        return self.replace(vlm_error=error)

    def with_open3d_verification(self, status: str, attempt_count: int) -> "FloorSelection":
        """Return this audit state with a completed Open3D verification update."""

        status_code = {
            "in_progress": 1,
            "accepted": 2,
            "unavailable": 3,
            "rejected": 4,
            "max_attempts_exhausted": 4,
        }[status]
        return self.replace(
            open3d_verification=self.open3d_verification.replace(
                status=jnp.asarray(status_code, dtype=jnp.int32),
                attempt_count=jnp.asarray(attempt_count, dtype=jnp.int32),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        method = _METHOD_NAMES[int(self.method)]
        verification = _VERIFICATION_NAMES[int(self.open3d_verification.status)]
        return {
            "method": method,
            "plane_index": int(self.plane_index),
            "vlm_error": self.vlm_error,
            "upright_flip": {
                "applied": bool(self.upright_flip_applied),
                "below_ratio": float(self.upright_below_ratio),
            },
            "wall_alignment": {
                "applied": bool(self.wall_alignment.applied),
                "plane_index": int(self.wall_alignment.plane_index),
                "normal": [float(value) for value in self.wall_alignment.normal],
                "inliers": int(self.wall_alignment.inliers),
                "yaw_degrees": float(self.wall_alignment.yaw_degrees),
                "perpendicular_tolerance_degrees": float(self.wall_alignment.perpendicular_tolerance_degrees),
            },
            "centering": {
                "source_centroid": [float(value) for value in self.centering.source_centroid],
                "translation": [float(value) for value in self.centering.translation],
                "target_xy": [0.0, 0.0],
                "floor_z": 0.0,
            },
            "open3d_verification": {
                "status": verification,
                "attempt_count": int(self.open3d_verification.attempt_count),
            },
        }

    # Host-side compatibility for existing consumers; it does not participate
    # in the PyTree representation.
    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


@struct.dataclass
class FloorAlignment:
    planes: PlaneSet
    floor_index: jax.Array
    rotation: jax.Array
    rotation_xyzw: jax.Array
    translation: jax.Array
    source_points: jax.Array
    selection: FloorSelection

    @property
    def floor(self) -> PlaneSummary:
        return self.planes[self.floor_index]

    def transform_points(self, points: jax.Array) -> jax.Array:
        """Express one point or a point batch in this alignment's map frame."""

        return points @ self.rotation.T + self.translation

    def transform_pose(self, pose: Pose) -> Pose:
        """Express one pose in this alignment's map frame."""

        return pose.transformed(self)

    def transform_poses(self, poses: Pose) -> Pose:
        """Transform a variable-length pose PyTree with fixed-size JAX batches.

        ``map_batches`` handles an empty ``poses`` itself (one empty batch,
        no padding), so there is no separate empty-input guard here.
        """

        return map_batches(
            _transform_poses_batch_jax,
            poses,
            self,
            in_axes=(0, None),
            batch_size=_JAX_BATCH_SIZE,
        )

    def upright_from_point_support(
        self, points: jax.Array, config: PlaneDetectionConfig
    ) -> "FloorAlignment":
        """Return this alignment with the map put *above* its selected floor.

        A fitted plane has two normals, and levelling against the wrong one is a
        perfectly valid transform that leaves the scene hanging below its own
        floor -- the floor then behaves like a ceiling. The selection pathway
        cannot see this: it chooses a *plane*, and both signs describe the same
        plane. So this reconsiders nothing about which plane was chosen and only
        looks at where the rest of the cloud ended up.

        The plane's own inliers (within ``distance_tolerance`` of z=0 in the
        aligned frame) are segmented out first, since they sit on both sides of
        zero by construction and would only dilute the vote. If essentially all
        of what remains is underneath -- ``upright_below_fraction`` -- the cloud
        is rotated 180 degrees about X, which keeps the floor at z=0 and keeps it
        level while moving everything else above it.

        Must run before ``align_longest_wall_and_center``: that step picks a yaw
        from a wall normal expressed in *this* frame and then commits the
        centroid translation, so flipping afterwards would invalidate both.
        """

        from traj_align.common.geometry import _matrix_to_quaternion_jax

        heights = (points @ self.rotation.T + self.translation)[:, 2]
        off_plane = jnp.abs(heights) > config.distance_tolerance
        off_plane_count = jnp.sum(off_plane)
        below_ratio = jnp.sum(off_plane & (heights < 0.0)) / jnp.maximum(off_plane_count, 1)
        # A map whose every point lies on the selected plane has nothing to be
        # above or below, and must not be flipped on a 0/0 vote.
        inverted = (off_plane_count > 0) & (below_ratio >= config.upright_below_fraction)
        rotation = jnp.where(inverted, _X_FLIP @ self.rotation, self.rotation)
        return self.replace(
            rotation=rotation,
            rotation_xyzw=_matrix_to_quaternion_jax(rotation),
            # The translation is part of the same map-frame transform, so it is
            # flipped with the rotation rather than recomputed; the floor stays
            # at z=0 either way.
            translation=jnp.where(inverted, _X_FLIP @ self.translation, self.translation),
            selection=self.selection.with_upright_flip(inverted, below_ratio),
        )

    def align_longest_wall_and_center(
        self, points: jax.Array, config: PlaneDetectionConfig
    ) -> "FloorAlignment":
        """Return this alignment with its deterministic yaw and XY-centering choice applied."""

        normals = self.planes.normals
        floor_normal = self.floor.normal
        perpendicular_limit = jnp.sin(jnp.deg2rad(config.wall_perpendicular_tolerance_degrees))
        candidate = self.planes.valid & (jnp.arange(normals.shape[0]) != self.floor_index)
        candidate &= jnp.abs(normals @ floor_normal) <= perpendicular_limit
        extent_area = jnp.prod(self.planes.extents, axis=1)
        # Inlier count is primary; the tiny terms only make ties deterministic.
        score = (
            self.planes.inlier_counts.astype(points.dtype)
            + extent_area * 1e-6
            + self.planes.structural_scores * 1e-9
        )
        wall_index = jnp.argmax(jnp.where(candidate, score, -jnp.inf)).astype(jnp.int32)
        wall = self.planes[wall_index]
        aligned_normal = self.rotation @ wall.normal
        horizontal_normal = aligned_normal.at[2].set(0.0)
        horizontal_norm = jnp.linalg.norm(horizontal_normal)
        applies = jnp.any(candidate) & (horizontal_norm > _JAX_EPSILON)
        horizontal_normal = horizontal_normal / jnp.maximum(horizontal_norm, _JAX_EPSILON)
        wall_direction = jnp.cross(_Z_AXIS, horizontal_normal)
        yaw = -jnp.arctan2(wall_direction[1], wall_direction[0])
        yaw = jnp.mod(yaw + jnp.pi / 2.0, jnp.pi) - jnp.pi / 2.0
        yaw_rotation = Rotation.from_euler("z", yaw).as_matrix()
        rotation = jnp.where(applies, yaw_rotation @ self.rotation, self.rotation)
        centroid = jnp.mean(points, axis=0)
        rotated_centroid = rotation @ centroid
        translation = jnp.array(
            (-rotated_centroid[0], -rotated_centroid[1], self.translation[2])
        )
        wall_selection = WallAlignment(
            applied=applies,
            plane_index=jnp.where(applies, wall_index, -1),
            normal=jnp.where(applies, wall.normal, jnp.zeros(3, dtype=points.dtype)),
            inliers=jnp.where(applies, wall.inliers, 0),
            yaw_degrees=jnp.where(applies, jnp.rad2deg(yaw), 0.0),
            perpendicular_tolerance_degrees=jnp.asarray(config.wall_perpendicular_tolerance_degrees),
        )
        return self.with_wall_alignment_and_centering(
            rotation=rotation,
            translation=translation,
            wall_alignment=wall_selection,
            centering=MapCentering(source_centroid=centroid, translation=translation),
        )

    @classmethod
    def estimate(
        cls,
        key: jax.Array,
        points: jax.Array,
        config: PlaneDetectionConfig | None = None,
    ) -> "FloorAlignment":
        """Detect structural planes and construct the geometry-selected alignment."""

        from traj_align.common.geometry import (
            _detect_plane_summaries,
            _heuristic_floor,
            _sample_points_jax,
        )

        config = config or PlaneDetectionConfig()
        if not isinstance(points, jax.Array):
            raise TypeError("estimate_floor_alignment requires a jax.Array; upload data in the host adapter.")
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError("Floor detection points must have shape (N, 3) with N >= 3.")
        if not bool(jnp.all(jnp.isfinite(points))):
            raise ValueError("Floor detection points must all be finite.")

        key, rng = jax.random.split(key)
        sampled = _sample_points_jax(rng, points, config.sample_points)
        key, rng = jax.random.split(key)
        planes = _detect_plane_summaries(rng, sampled, config)
        if not bool(jnp.any(planes.valid)):
            raise ValueError("Floor detection could not find any planes in the map points.")
        floor_index = _heuristic_floor(planes, config.structural_candidates)
        return cls.for_floor(floor_index, planes, points.shape[0], FloorSelection.geometry(floor_index))

    @classmethod
    def for_floor(
        cls,
        floor: PlaneSummary | int | jax.Array,
        planes: PlaneSet,
        source_points: int | jax.Array,
        selection: FloorSelection | Mapping[str, Any] | None = None,
    ) -> "FloorAlignment":
        """Build the exact level transform for one detected floor plane."""

        # Lazy import: geometry imports FloorAlignment (this class) at module
        # level, so importing geometry back at *this* module's top level
        # would be circular. Deferring to call time, exactly as
        # ``estimate()`` already does below for other geometry helpers, lets
        # this reuse geometry's ``_matrix_to_quaternion_jax`` instead of
        # duplicating the ``Rotation.from_matrix(...).as_quat()`` expression.
        from traj_align.common.geometry import _matrix_to_quaternion_jax

        floor_index = floor.index if isinstance(floor, PlaneSummary) else jnp.asarray(floor, dtype=jnp.int32)
        selected_floor = planes[floor_index]
        rotation = _rotation_between_jax(selected_floor.normal, _Z_AXIS)
        return cls(
            planes=planes,
            floor_index=jnp.asarray(floor_index, dtype=jnp.int32),
            rotation=rotation,
            rotation_xyzw=_matrix_to_quaternion_jax(rotation),
            translation=jnp.asarray((0.0, 0.0, selected_floor.offset)),
            source_points=jnp.asarray(source_points, dtype=jnp.int32),
            selection=FloorSelection.from_value(selection, floor_index),
        )

    def with_selected_floor(self, floor_index: int | jax.Array, method: int) -> "FloorAlignment":
        """Return this alignment rebuilt from a different detected floor plane."""

        floor_index = jnp.asarray(floor_index, dtype=jnp.int32)
        return self.for_floor(
            floor_index,
            self.planes,
            self.source_points,
            self.selection.with_selected_plane(floor_index, method),
        )

    def with_wall_alignment_and_centering(
        self,
        *,
        rotation: jax.Array,
        translation: jax.Array,
        wall_alignment: WallAlignment,
        centering: MapCentering,
    ) -> "FloorAlignment":
        """Return this alignment after applying its yaw and XY-centering decision."""

        # Lazy import for the same circular-import reason as ``for_floor`` above.
        from traj_align.common.geometry import _matrix_to_quaternion_jax

        return self.replace(
            rotation=rotation,
            rotation_xyzw=_matrix_to_quaternion_jax(rotation),
            translation=translation,
            selection=self.selection.with_wall_alignment(wall_alignment, centering),
        )

    def with_vlm_error(self, error: str) -> "FloorAlignment":
        """Return this alignment carrying the reason its VLM never chose a floor."""

        return self.replace(selection=self.selection.with_vlm_error(error))

    def with_open3d_verification(self, status: str, attempt_count: int) -> "FloorAlignment":
        """Return this alignment with its Open3D verification audit updated."""

        return self.replace(selection=self.selection.with_open3d_verification(status, attempt_count))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete alignment once for metadata and audits."""

        return {
            "floor": self.floor.to_dict(),
            "planes": [plane.to_dict() for plane in self.planes],
            "rotation": [[float(value) for value in row] for row in self.rotation],
            "rotation_xyzw": [float(value) for value in self.rotation_xyzw],
            "translation": [float(value) for value in self.translation],
            "source_points": int(self.source_points),
            "selection": self.selection.to_dict(),
        }


@jax.jit
def _transform_poses_batch_jax(poses: Pose, alignment: FloorAlignment) -> Pose:
    """Transform one fixed-size pose batch without host round trips."""

    return jax.vmap(alignment.transform_pose)(poses)


@struct.dataclass
class FloorAlignedArtifacts:
    metadata: Path = struct.field(pytree_node=False)
    observations: Path | None = struct.field(pytree_node=False)
    keyframe_trajectory: Path | None = struct.field(pytree_node=False)
    camera_trajectory: Path | None = struct.field(pytree_node=False)
    map_points: Path | None = struct.field(pytree_node=False)
    pointcloud: Path | None = struct.field(pytree_node=False)
    alignment: FloorAlignment


@struct.dataclass
class _RansacParameters:
    """Dynamic JAX leaves consumed by the compiled RANSAC core."""

    distance_tolerance: jax.Array
    min_inlier_ratio: jax.Array
    min_inliers: jax.Array

    @classmethod
    def from_config(cls, config: PlaneDetectionConfig) -> "_RansacParameters":
        """Build device-resident RANSAC parameters from host configuration."""

        return cls(
            distance_tolerance=jnp.asarray(config.distance_tolerance),
            min_inlier_ratio=jnp.asarray(config.min_inlier_ratio),
            min_inliers=jnp.asarray(config.min_inliers, dtype=jnp.int32),
        )


@struct.dataclass
class _RansacState:
    """Fixed-shape RANSAC carry state for ``lax.fori_loop``."""

    active: jax.Array
    random_key: jax.Array
    normals: jax.Array
    offsets: jax.Array
    valid: jax.Array

    @classmethod
    def initial(
        cls, random_key: jax.Array, point_count: int, max_planes: int, dtype: Any
    ) -> "_RansacState":
        """Create the empty fixed-shape carry state for a RANSAC run."""

        return cls(
            active=jnp.ones((point_count,), dtype=bool),
            random_key=random_key,
            normals=jnp.zeros((max_planes, 3), dtype=dtype),
            offsets=jnp.zeros((max_planes,), dtype=dtype),
            valid=jnp.zeros((max_planes,), dtype=bool),
        )

    def with_hypothesis(
        self,
        *,
        index: int,
        random_key: jax.Array,
        normal: jax.Array,
        offset: jax.Array,
        accepted: jax.Array,
        inlier_mask: jax.Array,
    ) -> "_RansacState":
        """Return the next carry state after evaluating one plane hypothesis."""

        return self.replace(
            active=jnp.where(accepted, self.active & ~inlier_mask, self.active),
            random_key=random_key,
            normals=self.normals.at[index].set(normal),
            offsets=self.offsets.at[index].set(offset),
            valid=self.valid.at[index].set(accepted),
        )


@struct.dataclass
class _ObservationView:
    index: int = struct.field(pytree_node=False)
    image: Path = struct.field(pytree_node=False)
    depth: Path | None = struct.field(pytree_node=False)
    pose: Pose


@struct.dataclass
class _PlaneVisualSample:
    observation_index: int = struct.field(pytree_node=False)
    image: Path = struct.field(pytree_node=False)
    score: float = struct.field(pytree_node=False)
    pixels: tuple[tuple[float, float], ...] = struct.field(pytree_node=False)


@struct.dataclass
class _PlaneContactSheet:
    plane_index: int = struct.field(pytree_node=False)
    path: Path = struct.field(pytree_node=False)
    samples: tuple[_PlaneVisualSample, ...] = struct.field(pytree_node=False)


@struct.dataclass
class _VlmFloorChoice:
    plane_index: int = struct.field(pytree_node=False)
    response: str = struct.field(pytree_node=False)
    parsed: dict[str, Any] = struct.field(pytree_node=False)
    artifacts: tuple[str, ...] = struct.field(pytree_node=False)


@struct.dataclass
class _VlmFloorResult:
    """The decisive floor answer, and every attempt it took to get one.

    ``attempts`` holds one record per repair round -- the raw reply, what was wrong
    with it, and the clause the next prompt carried -- for the same reason
    :class:`_VlmFloorElimination` does. It matters more here: this is the answer
    whose absence fails the build under ``vlm_required``, so "the model was asked
    three times and misread the question three times" and "the model was asked once"
    are two very different post-mortems, and without this field they leave the same
    record. A successful first attempt records a single entry, so the field is also
    how a reader tells a clean answer from a repaired one.
    """

    choice: _VlmFloorChoice | None = struct.field(pytree_node=False)
    error: str | None = struct.field(pytree_node=False, default=None)
    response: str = struct.field(pytree_node=False, default="")
    parsed: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)
    response_diagnostics: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)
    attempts: tuple[dict[str, Any], ...] = struct.field(pytree_node=False, default=())


@struct.dataclass
class _VlmFloorElimination:
    """One elimination turn's outcome: which candidates leave the ballot.

    Deliberately not a ``_VlmFloorResult``. That type carries a *decision* whose
    absence is a build failure under ``vlm_required``; this one carries a
    *narrowing*, whose absence means the next turn simply sees a wider ballot. Two
    types keep ``error`` from meaning two different things -- here it is never
    fatal, and ``traj_align.floor.vlm_images`` relies on that.

    ``eliminated`` is what was applied after sanitising (on the live ballot, no
    duplicates, no more than the turn asked for); ``answered`` is every integer the
    model actually named, so an audit reader can see the difference. ``attempts``
    holds one record per repair round -- the raw reply, what was wrong with it and
    the clause added to the next prompt -- because under greedy decoding a retry is
    only meaningful if the prompt changed, and that is worth being able to check.
    """

    eliminated: tuple[int, ...] = struct.field(pytree_node=False, default=())
    answered: tuple[int, ...] = struct.field(pytree_node=False, default=())
    attempts: tuple[dict[str, Any], ...] = struct.field(pytree_node=False, default=())
    discarded: str | None = struct.field(pytree_node=False, default=None)
    error: str | None = struct.field(pytree_node=False, default=None)
    parsed: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)
    response: str = struct.field(pytree_node=False, default="")
    response_diagnostics: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)


@struct.dataclass
class _VlmAlignmentVerification:
    accepted: bool = struct.field(pytree_node=False)
    plane_index: int | None = struct.field(pytree_node=False)
    response: str = struct.field(pytree_node=False)
    parsed: dict[str, Any] = struct.field(pytree_node=False)
    error: str | None = struct.field(pytree_node=False, default=None)
    response_diagnostics: dict[str, Any] = struct.field(pytree_node=False, default_factory=dict)
