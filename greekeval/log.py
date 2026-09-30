"""Logging helper (copied from greekllm.utils so the toolkit does not import the training package)."""
from __future__ import annotations

import logging

LOG_FMT = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")


def get_logger(name: str = "greekeval") -> logging.Logger:
    """Child loggers (greekeval.x) propagate to the 'greekeval' parent, which owns the handlers."""
    parent = logging.getLogger("greekeval")
    if not parent.handlers:
        h = logging.StreamHandler()
        h.setFormatter(LOG_FMT)
        parent.addHandler(h)
        parent.setLevel(logging.INFO)
        parent.propagate = False
    return logging.getLogger(name)
