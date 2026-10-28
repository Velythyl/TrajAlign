"""Provider-neutral spatial transforms used by alignment stages.

Floor alignment is intentionally richer than a transform: it contains plane
selection and verification evidence.  Scene alignment only needs the immutable
transform portion, so this module keeps that dependency out of the reusable
artifact and graph code.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

from traj_align.common.quaternion import matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix


class SpatialTransform(Protocol):
    """A transform that can be applied to an ``(N, 3)`` point array."""

    def transform_points(self, points: np.ndarray) -> np.ndarray: ...

    def transform_pose_dict(self, pose: dict[str, Any]) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class PlanarTransform:
    """A metric floor-frame transform: yaw plus translation in the XY plane."""

    yaw_radians: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in (self.yaw_radians, self.tx, self.ty)):
            raise ValueError("PlanarTransform values must be finite")

    @property
    def rotation(self) -> np.ndarray:
        cosine, sine = math.cos(self.yaw_radians), math.sin(self.yaw_radians)
        return np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation
        matrix[:2, 3] = (self.tx, self.ty)
        return matrix

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape[-1:] != (3,):
            raise ValueError("PlanarTransform expects points with final dimension 3")
        return values @ self.rotation.T + np.asarray((self.tx, self.ty, 0.0))

    def transform_pose_dict(self, pose: dict[str, Any]) -> dict[str, float]:
        return transform_pose_dict(pose, self)

    def inverse(self) -> "PlanarTransform":
        inverse_rotation = self.rotation.T
        translation = -inverse_rotation @ np.asarray((self.tx, self.ty, 0.0))
        return PlanarTransform(-self.yaw_radians, float(translation[0]), float(translation[1]))

    def compose(self, right: "PlanarTransform") -> "PlanarTransform":
        """Return ``self(right(point))``."""

        translation = self.transform_points(np.asarray(((right.tx, right.ty, 0.0),)))[0]
        return PlanarTransform(
            self.yaw_radians + right.yaw_radians,
            float(translation[0]),
            float(translation[1]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "se2",
            "yaw_radians": self.yaw_radians,
            "translation": [self.tx, self.ty, 0.0],
            "matrix": self.matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanarTransform":
        translation = value.get("translation", (0.0, 0.0, 0.0))
        return cls(float(value.get("yaw_radians", 0.0)), float(translation[0]), float(translation[1]))


def fit_planar_correspondences(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> PlanarTransform:
    """Closed-form weighted 2D Kabsch fit: the floor-preserving SE(2) that best maps ``source`` onto ``target``.

    Only yaw and XY translation are estimated -- floor alignment has already
    fixed roll, pitch and height for every trajectory independently
    (``traj_align.floor.types.FloorAlignment.align_longest_wall_and_center``),
    and the maps are metric (floor alignment is rigid; scale comes from the
    mapper, and ``align_scene_artifacts`` refuses maps declared ``arbitrary``),
    so the vertical axis is common knowledge, not something a scene-alignment
    method needs to (re-)discover. Two correspondences already fully
    determine this transform, unlike a general 6-DoF rigid fit, which needs
    three -- callers doing RANSAC over correspondences should sample minimal
    sets of two, not three (see ``ragmap.alignment.scene.vpr._verify_rgbd``
    and ``ragmap.alignment.scene.icp._global_registration_seed``).

    Shared by :func:`ragmap.alignment.scene.icp._fit_planar` (its JAX
    sibling, used inside the jitted local-ICP refinement loop) and
    :mod:`ragmap.alignment.scene.vpr` (RGB-D RANSAC correspondences). Keep
    all three in step if the formula changes -- see
    ``tests/test_quaternion_consolidation.py``.
    """

    weight = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    total = max(float(np.sum(weight)), 1.0)
    source_center = np.sum(source[:, :2] * weight[:, None], axis=0) / total
    target_center = np.sum(target[:, :2] * weight[:, None], axis=0) / total
    left, right = source[:, :2] - source_center, target[:, :2] - target_center
    cross = float(np.sum(weight * (left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0])))
    dot = float(np.sum(weight * (left[:, 0] * right[:, 0] + left[:, 1] * right[:, 1])))
    yaw = math.atan2(cross, dot)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    translation = target_center - np.asarray(
        ((cosine * source_center[0]) - (sine * source_center[1]), (sine * source_center[0]) + (cosine * source_center[1]))
    )
    return PlanarTransform(yaw, float(translation[0]), float(translation[1]))


def transform_pose_dict(pose: dict[str, Any], transform: PlanarTransform) -> dict[str, float]:
    """Apply an SE(2) transform to RAGMAP's xyzw pose dictionary."""

    result = {key: float(pose[key]) for key in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")}
    result["tx"], result["ty"], result["tz"] = transform.transform_points(
        np.asarray(((result["tx"], result["ty"], result["tz"]),))
    )[0].tolist()
    half_yaw = transform.yaw_radians / 2.0
    # q_yaw * q_pose, both expressed as xyzw.
    x, y, z, w = result["qx"], result["qy"], result["qz"], result["qw"]
    sine, cosine = math.sin(half_yaw), math.cos(half_yaw)
    result["qx"] = cosine * x - sine * y
    result["qy"] = cosine * y + sine * x
    result["qz"] = cosine * z + sine * w
    result["qw"] = cosine * w - sine * z
    return result


@dataclass(frozen=True, slots=True)
class AttachmentTransform:
    """The cross-floor counterpart of :class:`PlanarTransform`: yaw + 3D translation.

    An attachment relates two trajectories that floor alignment has already
    normalized *independently* -- each has its own z=0 reference, and unlike
    :func:`fit_planar_correspondences`'s coplanar case, nothing guarantees the
    two agree. Rotation stays 2D (about the shared vertical axis both sides
    independently established), so only the translation gains a z component:
    4-DOF, not a general 6-DOF rigid fit. See
    ``plans/attachment_stage.md`` and
    ``scripts/investigate_attachments.py`` (this type's original,
    standalone-script home) for the full argument.
    """

    planar: PlanarTransform
    tz: float

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        out = self.planar.transform_points(points)
        out[:, 2] += self.tz
        return out

    def inverse(self) -> "AttachmentTransform":
        return AttachmentTransform(self.planar.inverse(), -self.tz)

    def to_dict(self) -> dict[str, Any]:
        return {**self.planar.to_dict(), "tz": self.tz}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AttachmentTransform":
        return cls(PlanarTransform.from_dict(value), float(value.get("tz", 0.0)))


def fit_attachment_correspondences(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None
) -> AttachmentTransform:
    """Closed-form 4-DOF fit: :func:`fit_planar_correspondences` plus a ``tz`` offset.

    The planar (yaw + xy) part is fit exactly as it is for the coplanar
    (scene_alignment) case; ``tz`` is then the weighted mean of the residual
    z left over after applying that planar transform. Two correspondences
    still fully determine this transform -- RANSAC callers should keep
    sampling minimal sets of two, same as ``fit_planar_correspondences``.
    """

    planar = fit_planar_correspondences(source, target, weights)
    transformed = planar.transform_points(source)
    residual_z = target[:, 2] - transformed[:, 2]
    weight = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    tz = float(np.average(residual_z, weights=weight))
    return AttachmentTransform(planar, tz)


@dataclass(frozen=True, slots=True)
class RigidTransform:
    """Validated SE(3) transform for replaying an existing floor transform.

    Generic multi-session alignment is always :class:`PlanarTransform`; this
    type exists only for faithfully applying a previously selected full floor
    transform (not for estimating a new cross-trajectory transform).
    """

    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    translation: tuple[float, float, float]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.rotation, dtype=float)
        translation = np.asarray(self.translation, dtype=float)
        if matrix.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(matrix).all() or not np.isfinite(translation).all():
            raise ValueError("RigidTransform requires finite 3x3 rotation and three-vector translation")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
            raise ValueError("RigidTransform rotation must be orthonormal with determinant one")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float)
        if values.shape[-1:] != (3,):
            raise ValueError("RigidTransform expects points with final dimension 3")
        return values @ np.asarray(self.rotation).T + np.asarray(self.translation)

    def transform_pose_dict(self, pose: dict[str, Any]) -> dict[str, float]:
        result = {key: float(pose[key]) for key in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")}
        result["tx"], result["ty"], result["tz"] = self.transform_points(
            np.asarray(((result["tx"], result["ty"], result["tz"]),))
        )[0].tolist()
        current = quaternion_xyzw_to_matrix(result["qx"], result["qy"], result["qz"], result["qw"])
        rotated = np.asarray(self.rotation) @ current
        result["qx"], result["qy"], result["qz"], result["qw"] = matrix_to_quaternion_xyzw(rotated)
        return result

    @classmethod
    def from_floor_alignment(cls, value: dict[str, Any]) -> "RigidTransform":
        return cls(
            tuple(tuple(float(entry) for entry in row) for row in value["rotation"]),
            tuple(float(entry) for entry in value["translation"]),
        )
