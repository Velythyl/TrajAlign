"""Deterministic evenly-spaced subsampling, shared across mapping/alignment.

Five call sites (``ragmap.mapping.vggt`` x2, ``ragmap.mapping.viewer``,
``ragmap.alignment.scene.icp``, ``traj_align.floor.vlm_images``)
independently reimplemented the same "evenly spaced index" trick for
uniformly subsampling a point cloud, frame list, or generic sequence down to
a maximum size. This module is the single source of truth for that index
computation; it is pure NumPy so it stays safe to import from any of those
modules regardless of whether they otherwise depend on JAX or torch.
"""
from __future__ import annotations

import numpy as np


def uniform_subsample_indices(count: int, limit: int) -> np.ndarray:
    """Evenly spaced ``int64`` indices into ``range(count)``, capped at ``limit``.

    If ``limit`` is non-positive, or ``count`` already fits within it, every
    index ``0..count-1`` is returned unchanged (no subsampling needed).
    Otherwise returns exactly ``limit`` indices spanning ``0..count-1``
    inclusive via ``np.linspace``, matching each former call site's formula.
    """

    if limit <= 0 or count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, limit, dtype=np.int64)
