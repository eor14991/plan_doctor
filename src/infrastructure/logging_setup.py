"""
Infrastructure: Structured JSON Logging
Replaces all print() and ad-hoc logging.getLogger() calls in the original.
DEFECT #11 FIX: `print(f"UPLOAD PATH: {file_path}")` → logger.debug(extra={...})
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from pythonjsonlogger.json import JsonFormatter  # type: ignore[import]


def setup_logging(log_level: str = "INFO", service_name: Optional[str] = None) -> None:
    fmt = "%(timestamp)s %(level)s %(name)s %(funcName)s %(lineno)d %(message)s"
    formatter = JsonFormatter(
        fmt=fmt, rename_fields={"levelname": "level", "asctime": "timestamp"}, timestamp=True
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if root.hasHandlers():
        root.handlers.clear()
    root.addHandler(handler)
    for noisy in [
        "motor",
        "pymongo",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "transformers",
        "qdrant_client",
        "urllib3",
        "asyncio",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if service_name:

        class _Filter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                record.service = service_name  # type: ignore[attr-defined]
                return True

        root.addFilter(_Filter())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
