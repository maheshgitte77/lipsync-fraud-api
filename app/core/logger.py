"""Structured logging helper. One-line setup, reusable per-module loggers."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from app.core.config import settings

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = getattr(logging, settings.log_level, logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            h.setFormatter(logging.Formatter(fmt))
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(handler)
    root.setLevel(level)
    # Quiet known-noisy libs
    for noisy in ("urllib3", "kafka", "kafka.conn", "kafka.client", "botocore", "s3transfer"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name or settings.app_name)
