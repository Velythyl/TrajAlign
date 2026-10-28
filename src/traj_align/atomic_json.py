"""Crash-safe JSON writes: write to a private temp file, then atomically replace.

Extracted from ~20 call sites across mappers, floor/scene alignment, and
augmenters (Phase 4 of ``plans/refactor_and_cleanup.md``) that each did
``destination.write_text(json.dumps(payload) + "\\n")`` directly. A process
killed mid-``write_text`` (OOM-killed GPU step, SIGKILL escalation from
``process.py``, a cancelled SLURM step) leaves a truncated file at
*destination* -- and every one of these files is ``json.loads`` back by a
later stage (``map.json`` by floor alignment, ``scene_alignment.json`` by
augmentation, ``floor_alignment.json`` by the native-Atlas identity path,
...). ``os.replace``/``Path.replace`` is atomic on the same filesystem, so
writing the full payload to a temp file first and renaming it over
*destination* only afterward means *destination* is always either the
previous complete version or the new complete version, never a partial one.

The temp filename embeds both the PID and the thread id, not just a fixed
``.tmp`` suffix. Two different *processes* can legitimately be writing the
same trajectory's artifacts at once -- concurrent SLURM steps for one
trajectory exist under the durable path (e.g. a retried task racing a
not-yet-reaped previous attempt) -- and ``progress.py``'s
``ProgressReporter`` can be driven from more than one thread inside a single
process. A fixed suffix lets two writers' temp files collide, so one writer's
in-progress content can be renamed over the *other* writer's still-being-written
temp file, or one writer can unlink a temp file the other is mid-write to.
PID+thread-id makes every writer's temp file exclusively its own.

This module never widens what a caller catches. It raises whatever
``Path.write_text``/``Path.replace`` themselves raise (``OSError``) or
whatever ``json.dumps`` raises on a bad payload (``TypeError``); it does not
wrap those in a different exception type. That matters because
``docs/progress-reporting.md`` documents timing as "best-effort telemetry: a
reporting failure must not fail a mapper", and the one caller that relies on
that today (``progress.py``'s ``write_stage_timing_log``) wraps its
call in ``except OSError: pass``. If this helper raised something else on a
transient disk hiccup, that guard would stop working and a reporting failure
would fail the GPU step it is only supposed to be timing.

Kept stdlib-only: ``job_registry`` imports this transitively, and per
``step_runner.py`` every Celery launcher must never pull in torch/JAX.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


def write_json_atomic(
    destination: Path,
    payload: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
) -> None:
    """Write *payload* as JSON to *destination* without ever leaving a partial file.

    Serializes first, writes the whole result to a PID+thread-scoped temp
    file beside *destination*, then ``Path.replace``s it into place -- the
    one atomic step. If serialization or either filesystem operation raises,
    the temp file is removed (best-effort; a failure to clean it up is not
    re-raised over the original error) and the original exception propagates
    unchanged, so callers keep whatever ``except`` clause they already wrote
    around the old direct ``write_text`` call.
    """

    destination = Path(destination)
    text = json.dumps(payload, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
    temporary = destination.with_name(
        f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
