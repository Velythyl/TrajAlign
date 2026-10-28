"""Floor-alignment diagnostic logging."""
from __future__ import annotations
import logging
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

LOGGER = logging.getLogger("traj_align.floor")
_ACTIVE_LOG_DESTINATION: ContextVar[Path | None] = ContextVar("floor_log_destination", default=None)


class _AlignmentLogFilter(logging.Filter):
    def __init__(self, destination: Path) -> None:
        super().__init__()
        self.destination = destination

    def filter(self, record: logging.LogRecord) -> bool:
        return _ACTIVE_LOG_DESTINATION.get() == self.destination


def _format_normal(normal: tuple[float, float, float]) -> str:
    return "(" + ", ".join(f"{value:.4f}" for value in normal) + ")"


def _report_progress(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)

@contextmanager
def _floor_diagnostic_log(output_dir: Path) -> Iterable[Path]:
    """Mirror every alignment decision to an inspectable map-local log file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "floor_alignment.log"
    handler = logging.FileHandler(destination, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler.addFilter(_AlignmentLogFilter(destination))
    LOGGER.addHandler(handler)
    previous_level = LOGGER.level
    LOGGER.setLevel(logging.INFO)
    token = _ACTIVE_LOG_DESTINATION.set(destination)
    try:
        yield destination
    finally:
        _ACTIVE_LOG_DESTINATION.reset(token)
        LOGGER.removeHandler(handler)
        LOGGER.setLevel(previous_level)
        handler.close()
