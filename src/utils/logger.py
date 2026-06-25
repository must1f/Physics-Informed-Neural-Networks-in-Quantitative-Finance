"""Structured logging setup using loguru.

Removes loguru's default handler on import and installs a single console
handler that writes to ``stderr`` at INFO level with a compact
``HH:mm:ss | LEVEL | module | message`` format.

Usage::

    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("training started")
"""

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Console handler — concise format
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> | {message}",
    level="INFO",
)


def get_logger(name: str):
    """Return a loguru logger bound to a module name.

    The returned logger inherits the console handler configured at module
    import (INFO level, stderr). Use ``name=__name__`` at the call site
    so the log record's ``{name}`` field identifies the originating module.

    Args:
        name: Module or component identifier string (typically ``__name__``).

    Returns:
        A ``loguru.Logger`` instance pre-bound with ``name=name``.
    """
    return logger.bind(name=name)
