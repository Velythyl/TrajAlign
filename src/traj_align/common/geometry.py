"""RANSAC plane detection and deterministic floor-alignment heuristics."""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.spatial.transform import Rotation
from traj_align.floor.types import (
    FloorAlignment, _RansacParameters, _RansacState,
    _rotation_between_jax,  # noqa: F401 - retained as a geometry-module compatibility export.
)
from traj_align.common.types import PlaneDetectionConfig, PlaneSet

_JAX_EPSILON = 1e-9


@jax.jit
def _safe_quaternion_jax(quaternion: jax.Array) -> jax.Array:
    """Normalize xyzw input while mapping an invalid zero quaternion to identity."""

    norm = jnp.linalg.norm(quaternion)
    identity = jnp.array((0.0, 0.0, 0.0, 1.0), dtype=quaternion.dtype)
    return jnp.where(norm > _JAX_EPSILON, quaternion / norm, identity)


@jax.jit
def _quaternion_to_matrix_jax(quaternion: jax.Array) -> jax.Array:
    return Rotation.from_quat(_safe_quaternion_jax(quaternion)).as_matrix()


@jax.jit
def _matrix_to_quaternion_jax(matrix: jax.Array) -> jax.Array:
    return Rotation.from_matrix(matrix).as_quat()


@jax.jit
def _z_rotation_jax(angle: jax.Array) -> jax.Array:
    return Rotation.from_euler("z", angle).as_matrix()


@jax.jit
def _plane_from_triplet_jax(sample: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit one RANSAC hypothesis; degenerate triples are marked invalid."""

    normal = jnp.cross(sample[1] - sample[0], sample[2] - sample[0])
    norm = jnp.linalg.norm(normal)
    valid = norm > _JAX_EPSILON
    normal = normal / jnp.maximum(norm, _JAX_EPSILON)
    return normal, -jnp.dot(normal, sample[0]), valid


@jax.jit
def _plane_inlier_mask_jax(
    points: jax.Array, normal: jax.Array, offset: jax.Array, tolerance: jax.Array
) -> jax.Array:
    return plane_distances_jax(points, normal, offset) <= tolerance


@jax.jit
def plane_distances_jax(points: jax.Array, normal: jax.Array, offset: jax.Array) -> jax.Array:
    """Vectorized plane distance used by geometry and visual-evidence kernels."""

    return jnp.abs(points @ normal + offset)


@jax.jit
def _multiply_quaternions_jax(left: jax.Array, right: jax.Array) -> jax.Array:
    return (Rotation.from_quat(_safe_quaternion_jax(left)) * Rotation.from_quat(
        _safe_quaternion_jax(right)
    )).as_quat()

def estimate_floor_alignment(
    key: jax.Array, points: jax.Array, config: PlaneDetectionConfig | None = None
) -> FloorAlignment:
    """Compatibility entry point for :meth:`FloorAlignment.estimate`."""

    return FloorAlignment.estimate(key, points, config)


def _sample_points_jax(key: jax.Array, points: jax.Array, limit: int) -> jax.Array:
    """Deterministically sample on device without a NumPy RNG round trip.

    Looks like ``vlm_images._sample_host_points``, and both pick a random,
    index-sorted subsample -- but they are not the same function wearing two
    hats. ``points`` here is a ``jax.Array`` that ``FloorAlignment.estimate``
    keeps device-resident end to end (into ``_detect_plane_summaries``), so
    this stays a JAX in/JAX out function and never touches the host.
    ``_sample_host_points`` takes and returns ``np.ndarray`` for a caller that
    is host-resident (image projection, PIL) apart from the random draw
    itself; forcing it through this function would add a full point-cloud
    host<->device round trip where today only the small index array crosses
    that boundary. That is a real cost difference, not a style choice, so
    the two are kept separate.
    """

    if points.shape[0] > limit:
        indices = jax.random.choice(key, points.shape[0], (limit,), replace=False)
        return points[jnp.sort(indices)]
    return points


@functools.partial(jax.jit, static_argnames=("max_planes", "iterations", "candidate_batch"))
def _detect_planes_core_jax(
    key: jax.Array,
    points: jax.Array,
    parameters: _RansacParameters,
    *,
    max_planes: int,
    iterations: int,
    candidate_batch: int = 32,
) -> _RansacState:
    """Sequential masked RANSAC with static shapes and bounded candidates."""

    count = points.shape[0]
    required = jnp.maximum(parameters.min_inliers, jnp.ceil(count * parameters.min_inlier_ratio).astype(jnp.int32))
    padded_iterations = ((iterations + candidate_batch - 1) // candidate_batch) * candidate_batch

    def best_hypothesis(key: jax.Array, active: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        probabilities = active.astype(points.dtype)
        probabilities = probabilities / jnp.maximum(probabilities.sum(), 1.0)
        sample_indices = jax.random.choice(key, count, (padded_iterations, 3), replace=True, p=probabilities)
        normals, offsets, valid = jax.vmap(_plane_from_triplet_jax)(points[sample_indices])
        normals = normals.reshape((-1, candidate_batch, 3))
        offsets = offsets.reshape((-1, candidate_batch))
        valid = valid.reshape((-1, candidate_batch))

        def count_block(block: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            block_normals, block_offsets, block_valid = block
            distances = jnp.abs(points @ block_normals.T + block_offsets)
            counts = jnp.sum((distances <= parameters.distance_tolerance) & active[:, None], axis=0)
            return jnp.where(block_valid, counts, -1)

        counts = lax.map(count_block, (normals, offsets, valid)).reshape((-1,))[:iterations]
        index = jnp.argmax(counts)
        flat_normals = normals.reshape((-1, 3))[:iterations]
        flat_offsets = offsets.reshape((-1,))[:iterations]
        return flat_normals[index], flat_offsets[index], counts[index]

    def body(index: int, state: _RansacState) -> _RansacState:
        key, rng = jax.random.split(state.random_key)
        normal, offset, hypothesis_count = best_hypothesis(
            rng, state.active
        )
        mask = state.active & _plane_inlier_mask_jax(points, normal, offset, parameters.distance_tolerance)
        centroid = jnp.sum(points * mask[:, None], axis=0) / jnp.maximum(mask.sum(), 1)
        centered = points - centroid
        covariance = (centered * mask[:, None]).T @ centered
        _, eigenvectors = jnp.linalg.eigh(covariance)
        fitted_normal = eigenvectors[:, 0]
        fitted_normal = fitted_normal / jnp.maximum(jnp.linalg.norm(fitted_normal), _JAX_EPSILON)
        fitted_offset = -jnp.dot(fitted_normal, centroid)
        accepted = hypothesis_count >= required
        normal = jnp.where(accepted, fitted_normal, jnp.zeros(3, dtype=points.dtype))
        offset = jnp.where(accepted, fitted_offset, 0.0)
        return state.with_hypothesis(
            index=index,
            random_key=key,
            normal=normal,
            offset=offset,
            accepted=accepted,
            inlier_mask=mask,
        )

    initial = _RansacState.initial(key, count, max_planes, points.dtype)
    return lax.fori_loop(0, max_planes, body, initial)


def _detect_plane_summaries(
    key: jax.Array, points: jax.Array, config: PlaneDetectionConfig
) -> PlaneSet:
    state = _detect_planes_core_jax(
        key,
        points,
        _RansacParameters.from_config(config),
        max_planes=config.max_planes,
        iterations=config.ransac_iterations,
    )
    return PlaneSet.from_ransac_state(points, state, config.distance_tolerance)


@jax.jit
def _heuristic_floor(summaries: PlaneSet, structural_candidates: int) -> jax.Array:
    candidate = (jnp.arange(summaries.valid.shape[0]) < structural_candidates) & summaries.valid
    normal_z = jnp.where(candidate, summaries.normals[:, 2], -jnp.inf)
    upward = candidate & (normal_z >= jnp.max(normal_z) - 1e-3)
    score = jnp.where(upward, summaries.structural_scores, -jnp.inf)
    return jnp.argmax(score).astype(jnp.int32)
