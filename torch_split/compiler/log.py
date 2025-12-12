"""Logging utilities with Rich-based colored output."""

import logging
import os

from rich.panel import Panel
from rich.console import Console
from rich.traceback import Traceback
from rich.logging import RichHandler

_console = Console()


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a logger configured with RichHandler."""
    logger = logging.getLogger(name)

    level_name = os.getenv("TORCHSPLIT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        # avoid duplicating handlers
        handler = RichHandler(
            console=Console(highlight=False, markup=True),
            rich_tracebacks=True,
            markup=True,
            highlighter=None,
            show_time=True,
            show_path=False,
        )
        handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def set_level(level: str):
    lvl = getattr(logging, level.upper(), None)
    if lvl is None:
        raise ValueError(f"Invalid log level: {level}")
    logging.getLogger().setLevel(lvl)


def log_exception(exc: Exception):
    """Print an exception inside a Rich-styled box with traceback."""

    tb = Traceback.from_exception(
        type(exc),
        exc,
        exc.__traceback__,
        show_locals=True,
        width=100,
    )

    panel = Panel(tb, title=f"[bold red] {type(exc).__name__}", border_style="red", padding=(1, 2))

    _console.print(panel)
