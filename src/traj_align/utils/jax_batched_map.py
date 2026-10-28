"""Fixed-shape, PyTree-aware execution helpers for variable-sized JAX work.

The helpers deliberately batch *outside* the compiled kernel.  A caller keeps
one module-level ``jax.jit(jax.vmap(...))`` kernel and passes it to
``iter_batches`` or ``map_batches``; every non-empty invocation of that kernel
then has the same leading shape, including the padded final batch. Empty
input is a legitimate shape too: :func:`iter_batches` yields exactly one
zero-length batch for it rather than raising, so callers do not each need
their own empty-input special case.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Sequence

import jax
import jax.numpy as jnp

# The single source of truth for RAGMAP's default fixed-size JAX batch. Import
# this rather than redefining ``2048`` locally -- it used to be duplicated in
# four modules (``alignment/common/artifacts.py``, ``alignment/floor/types.py``,
# ``alignment/floor/vlm_images.py``, ``alignment/floor/open3d.py``).
DEFAULT_BATCH_SIZE = 2_048


def _normalise_in_axes(in_axes: int | None | Sequence[int | None], count: int) -> tuple[int | None, ...]:
    if isinstance(in_axes, (int, type(None))):
        return (in_axes,) * count
    if len(in_axes) != count:
        raise ValueError("in_axes must provide one axis for every positional argument.")
    if any(axis not in (0, None) for axis in in_axes):
        raise ValueError("Only leading-axis mapping (0) and unmapped arguments (None) are supported.")
    return tuple(in_axes)


def _leading_size(value: Any) -> int:
    leaves = jax.tree.leaves(value)
    if not leaves:
        raise ValueError("A mapped argument must contain at least one array leaf.")
    size = leaves[0].shape[0]
    if any(leaf.ndim == 0 or leaf.shape[0] != size for leaf in leaves):
        raise ValueError("Every mapped array leaf must share a leading dimension.")
    return size


def _slice_and_pad(value: Any, start: int, stop: int, batch_size: int) -> Any:
    length = stop - start

    def slice_and_pad(array: Any) -> jax.Array:
        chunk = jnp.asarray(array[start:stop])
        if length < batch_size:
            chunk = jnp.concatenate((chunk, jnp.repeat(chunk[-1:], batch_size - length, axis=0)), axis=0)
        return chunk

    return jax.tree.map(slice_and_pad, value)


def _trim_batch(value: Any, length: int) -> Any:
    return jax.tree.map(lambda array: array[:length], value)


def pad_leading(value: Any, size: int) -> tuple[Any, jax.Array]:
    """Pad a PyTree's shared leading dimension up to a fixed ``size``.

    Returns ``(padded, valid)``: ``padded`` repeats each leaf's last row --
    the same convention :func:`iter_batches` uses for a batch's final,
    partial chunk -- until every leaf has leading dimension ``size``, and
    ``valid`` is a ``(size,)`` boolean marking which rows are real data
    versus padding.

    Padding never changes a result by itself; it only gives a JAX kernel one
    fixed shape regardless of ``value``'s native size, so shape-triggered
    recompilation does not scale with how many distinct sizes show up across
    calls. A caller must mask or weight every reduction by ``valid`` itself
    (see ``ragmap.alignment.scene.icp``, whose per-pair ICP shape otherwise
    forced a fresh XLA compile for every distinct point-cloud size).
    """

    length = _leading_size(value)
    if length > size:
        raise ValueError(f"Cannot pad a leading dimension of {length} up past the requested size {size}.")
    if length == 0:
        padded = jax.tree.map(lambda array: jnp.zeros((size, *array.shape[1:]), dtype=array.dtype), value)
    else:
        padded = _slice_and_pad(value, 0, length, size)
    return padded, jnp.arange(size) < length


def iter_batches(
    fn: Callable[..., Any],
    *args: Any,
    in_axes: int | None | Sequence[int | None],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[Any]:
    """Yield results from a pre-compiled fixed-size batch kernel.

    ``fn`` receives full batches, not individual elements.  Arguments marked
    with ``in_axes=0`` are sliced and padded; ``None`` arguments are passed to
    every invocation unchanged.  This lets callers keep transforms such as an
    alignment or camera pose dynamic without baking them into a closure.

    An empty mapped leading dimension yields exactly one batch of that same
    (zero) size, calling ``fn`` directly with no slicing or padding -- there
    is nothing to pad an empty batch to.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not args:
        raise ValueError("At least one argument is required.")
    axes = _normalise_in_axes(in_axes, len(args))
    mapped = [(argument, axis) for argument, axis in zip(args, axes, strict=True) if axis == 0]
    if not mapped:
        raise ValueError("At least one argument must use in_axes=0.")
    size = _leading_size(mapped[0][0])
    if any(_leading_size(argument) != size for argument, _ in mapped[1:]):
        raise ValueError("Every mapped argument must share a leading dimension.")
    if size == 0:
        yield fn(*args)
        return

    for start in range(0, size, batch_size):
        stop = min(start + batch_size, size)
        batch_args = tuple(
            _slice_and_pad(argument, start, stop, batch_size) if axis == 0 else argument
            for argument, axis in zip(args, axes, strict=True)
        )
        yield _trim_batch(fn(*batch_args), stop - start)


def map_batches(
    fn: Callable[..., Any],
    *args: Any,
    in_axes: int | None | Sequence[int | None],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Any:
    """Collect :func:`iter_batches` results into one PyTree on the device.

    ``iter_batches`` always yields at least one batch (an empty mapped
    leading dimension yields one empty batch), so there is always something
    to concatenate.
    """

    batches = tuple(iter_batches(fn, *args, in_axes=in_axes, batch_size=batch_size))
    return jax.tree.map(lambda *arrays: jnp.concatenate(arrays, axis=0), *batches)
