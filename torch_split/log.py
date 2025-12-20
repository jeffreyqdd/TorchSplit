"""Logging utilities with Rich-based colored output."""

import logging
import os
import threading
import time
import torch.cuda as cuda
from contextlib import contextmanager

import psutil
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.text import Text
from rich.traceback import Traceback


_console = Console(highlight=False)
_root_logger = logging.getLogger()
_root_logger.setLevel(getattr(logging, str(os.getenv("TORCHSPLIT_LOG_LEVEL")).upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a logger configured with RichHandler."""
    logger = _root_logger.getChild(name)
    logger.setLevel(logging.NOTSET)

    if logger.handlers:
        return logger

    handler = RichHandler(
        console=_console,
        rich_tracebacks=True,
        markup=True,
        highlighter=None,
        show_time=True,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
    logger.addHandler(handler)
    logger.propagate = True
    return logger


def set_level(level: str | int) -> None:
    """Set the logging level for the root logger and all active loggers."""
    if isinstance(level, str):
        lvl = getattr(logging, level.upper(), None)
        if lvl is None:
            raise ValueError(f"Invalid log level: {level}")
    else:
        lvl = level

    _root_logger.setLevel(lvl)


@contextmanager
def suppress_logs(level: str = "CRITICAL"):
    """Temporarily raise the log level to suppress verbose output.

    Args:
        level: Log level to set (e.g., "CRITICAL", "ERROR", "WARNING"). Default is "CRITICAL".

    Usage:
        with suppress_logs("ERROR"):
            # Logs below ERROR level will be suppressed here
            some_operation()
    """
    original_level = _root_logger.level

    set_level(level)
    try:
        yield
    finally:
        set_level(original_level)


def log_exception(exc: Exception) -> None:
    """Print an exception inside a Rich-styled box with traceback."""
    tb = Traceback.from_exception(
        type(exc),
        exc,
        exc.__traceback__,
        show_locals=True,
        width=100,
    )

    panel = Panel(
        tb,
        title=f"[bold red]{type(exc).__name__}",
        border_style="red",
        padding=(1, 2),
    )
    _console.print(panel)


class ResourcesColumn(ProgressColumn):
    """Renders the 'resources' field from task fields."""

    def render(self, task) -> Text:
        resources = task.fields.get("resources", "")
        return Text(resources, style="cyan", justify="right")


@contextmanager
def progress_bar(name: str, transient: bool = False, refresh_rate: float = 0.5, **kwargs):
    """Context manager for a progress bar with live resource monitoring."""
    refresh_rate = max(refresh_rate, 0.1)
    update_interval = 1.0 / refresh_rate
    stop_event = threading.Event()
    gpu_available = cuda.is_available() and cuda.device_count() > 0

    def monitor_resources(progress: Progress, task_id: TaskID) -> None:
        """Continuously collect and update system resource metrics."""
        while not stop_event.is_set():
            try:
                metrics = [f"DRAM: {psutil.virtual_memory().percent:.1f}%"]

                if gpu_available:
                    for device_id in range(cuda.device_count()):
                        free, total = cuda.mem_get_info(device=device_id)
                        used_mb = (total - free) >> 20
                        total_mb = total >> 20
                        metrics.append(f"GPU{device_id}: {used_mb}/{total_mb} MB")

                progress.update(task_id, resources=" | ".join(metrics))
            except Exception:
                pass  # Gracefully handle any rendering errors

            time.sleep(update_interval)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        ResourcesColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=transient,
    )

    task_id = progress.add_task(name, resources="DRAM: 0.0%", **kwargs)
    progress.start()
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(progress, task_id),
        daemon=True,
    )
    monitor_thread.start()

    try:
        yield progress, task_id
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
        progress.stop()
