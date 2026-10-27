"""Small, dependency-free helpers for reporting nested workflow timings."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator, Literal


#: ``failed`` is ``finished``'s counterpart, not a replacement for it: a stage
#: reports exactly one of the two, and both stop the clock.
#:
#: It exists because the timer stops in a ``finally`` -- it must, or a stage that
#: raised would stay ``running`` forever -- and so a stage that threw used to be
#: indistinguishable in ``stage_timings.json`` from one that succeeded. That is how
#: a floor step could die on its first VLM call, publish no aligned artifact, and
#: still leave ``{"name": "Floor alignment", "elapsed_seconds": 145.9,
#: "running": false}`` behind for every downstream reader to trust.
TimingEvent = Literal["started", "finished", "failed"]

#: ``(stage, event, elapsed_seconds, reason)``. ``reason`` is set only for
#: ``failed`` and is the exception rendered as ``"<Type>: <message>"``.
#:
#: Defaulted rather than required so the many callbacks that only care about
#: durations -- and the tests that pass a two-argument lambda -- keep working;
#: :func:`timed_stage` passes it positionally only when it has one to pass.
StageTimingCallback = Callable[..., None]


@contextmanager
def timed_stage(
    callback: StageTimingCallback | None, name: str
) -> Iterator[None]:
    """Report a named stage's elapsed wall-clock time and how it ended.

    The callback boundary deliberately only carries a name, an outcome and an
    elapsed time, making it suitable for mappers, augmenters, or any future
    pipeline stage without coupling those components to dashboard state.

    The exception is re-raised untouched; this only records that there was one.
    """

    if callback is not None:
        callback(name, "started", None)
    started = perf_counter()
    try:
        yield
    except BaseException as exc:
        if callback is not None:
            callback(
                name,
                "failed",
                perf_counter() - started,
                f"{type(exc).__name__}: {exc}",
            )
        raise
    else:
        if callback is not None:
            callback(name, "finished", perf_counter() - started)
