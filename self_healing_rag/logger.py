from __future__ import annotations

import logging
import os
import sys


class _CliOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.endswith(".cli")


class _NotCli(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.endswith(".cli")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure_logging() -> None:
    """Send CLI output to stdout and library logs to stderr.

    Set LOG_LEVEL=WARNING in .env or the shell to hide the retry-strategy logs.
    """
    root = logging.getLogger("self_healing_rag")
    if root.handlers:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))
    root.propagate = False

    cli_handler = logging.StreamHandler(sys.stdout)
    cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_handler.addFilter(_CliOnly())

    library_handler = logging.StreamHandler(sys.stderr)
    library_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    library_handler.addFilter(_NotCli())

    root.addHandler(cli_handler)
    root.addHandler(library_handler)
