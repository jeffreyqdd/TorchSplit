"""Logging utilities with Rich-based colored output."""

import contextlib
import functools
import logging
import threading
import time

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Get a logger with RichHandler for colored output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # logger should show time, but not the path to reduce clutter
        # name should give us more than enough information about the source
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )

        formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    logger.name = name
    return logger


def timer(logger=None, name=None):
    """Decorator to log the execution time of a function."""

    def decorator(func):
        log = logger or logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter() - start) * 1000
                log.info("%s took %.2f ms", name or func.__name__, elapsed)

        return wrapper

    return decorator


# Thread-local storage for tracking nested timed_execution calls
_timing_context = threading.local()


@contextlib.contextmanager
def timed_execution(name: str, logger=None):
    """Context manager that logs execution time of a code block.

    Only logs timing information on the outermost call to handle recursive calls properly.

    Args:
        name: Name of the operation being timed
        logger: Logger instance to use. If None, uses the root logger.

    Example:
        with timed_execution("Data processing"):
            # ... your code here ...
            process_data()
    """
    log = logger or logging.getLogger()

    # Initialize nesting level if not exists
    if not hasattr(_timing_context, "nesting_level"):
        _timing_context.nesting_level = 0

    # Track if this is the outermost call
    is_outermost = _timing_context.nesting_level == 0
    start = None

    if is_outermost:
        log.info("[cyan]Starting[/] %s", name)
        start = time.perf_counter()

    _timing_context.nesting_level += 1

    try:
        yield
    finally:
        _timing_context.nesting_level -= 1

        # Only log completion time on the outermost call
        if is_outermost and start is not None:
            elapsed = (time.perf_counter() - start) * 1000
            log.info("[green]Completed[/] %s [dim]in %.2f ms[/]", name, elapsed)
