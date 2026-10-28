"""NumPy quaternion <-> rotation-matrix conversions, xyzw order throughout.

This is the NumPy sibling of :mod:`traj_align.common.geometry`'s JAX
conversions (``_quaternion_to_matrix_jax`` / ``_matrix_to_quaternion_jax``).
The two modules implement the *same* formulas in two different array
libraries -- that split is deliberate (this module must stay JAX-free so
NumPy-only callers, and anything importing them, are not forced to pull in
JAX) -- so keep them numerically in step whenever either changes. See
``tests/test_quaternion_consolidation.py`` for the cross-check between them
and for the characterization tests that pinned this module's branch
predicate against adversarial rotation matrices before it replaced the three
independent implementations that used to live in
``traj_align.common.transforms``, ``ragmap.mapping.vggt``, and
``ragmap.alignment.scene.vpr``.

Quaternion component order is **xyzw** everywhere in this module (matching
RAGMAP's on-disk pose dict convention ``{qx, qy, qz, qw}``). Callers that hold
a **wxyz** tuple (e.g. :mod:`ragmap.mapping.poses`, :mod:`ragmap.retrieval`)
must reorder before calling in.
"""
from __future__ import annotations

import math

import numpy as np


def quaternion_xyzw_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix for a quaternion given in **xyzw** order.

    The quaternion is normalized internally, so a near-unit input (e.g. one
    that has been through a limited-precision text round trip) is handled
    gracefully. Raises ``ValueError`` on an exactly zero quaternion.

    This function always normalizes and always raises on zero -- callers
    inside a ``jax.jit`` region cannot do either (no exceptions, and a
    data-dependent branch would need ``jnp.where``), which is why the JAX
    sibling here is :func:`traj_align.common.geometry._safe_quaternion_jax`
    composed with ``Rotation.from_quat``, mapping zero to the identity
    quaternion instead of raising. That is a deliberate policy difference
    driven by the JIT constraint, not an oversight.
    """

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray((
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    ))


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Unit quaternion, in **xyzw** order, for a proper rotation matrix.

    Uses the numerically stable Shepperd's-method branch predicate: the
    ``trace > 0`` case, falling back to whichever diagonal entry is largest
    (via ``argmax``) when the trace is non-positive. This was verified
    (``tests/test_quaternion_consolidation.py``) to produce identical output
    -- including at exact diagonal ties, where the branch taken differs from
    an if/elif chain but the formula is mathematically the same quaternion
    either way -- to the independent if/elif-chain implementation that used
    to live in ``ragmap.mapping.vggt``, across identity, 180-degree,
    near-180-degree, trace-near-zero, and random-rotation inputs. It also
    round-trips (``quaternion_xyzw_to_matrix`` recovers the input matrix) on
    all of those inputs.
    """

    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale, 0.25 * scale)
    index = int(np.argmax(np.diag(matrix)))
    if index == 0:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        return (0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale)
    if index == 1:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        return ((matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale)
    scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    return ((matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale)
