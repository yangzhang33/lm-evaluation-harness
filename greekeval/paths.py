"""Where things are. Every path the toolkit touches is resolved here, so it runs the same from any directory.

  EVAL_DIR      the folder holding this package and the lm-eval fork
  HARNESS       the lm-eval fork; lm-eval is launched with this as its working directory
  PROJECT_ROOT  where models/, runs/, data/ and reports/ live: $GREEKEVAL_PROJECT, else the folder above EVAL_DIR
  proj(p)       an absolute path is returned as is, a relative one is taken from PROJECT_ROOT, never from the
                current directory

Paths are built with os.path.join/abspath only, never realpath: lm-eval names each output subfolder after the model
path string it was given, so expanding a symlink would put a second results json next to the existing one.
"""
from __future__ import annotations

import os

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(EVAL_DIR, "lm-eval-adapted/lm-evaluation-harness")
PROJECT_ROOT = os.path.abspath(os.environ.get("GREEKEVAL_PROJECT") or os.path.dirname(EVAL_DIR))


def proj(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
