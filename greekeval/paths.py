"""Where things are. Every path the toolkit touches is resolved here, so it runs the same from any directory.

  EVAL_DIR      the toolkit folder: this package, the lm-eval fork, config/
  HARNESS       the lm-eval fork; lm-eval is launched with this as its working directory
  CONFIG_DIR    EVAL_DIR/config: the plan (plan.yaml) and the model registry (models.yaml, git-ignored)
  REGISTRY      the model registry file: $GREEKEVAL_MODELS, else CONFIG_DIR/models.yaml
  PROJECT_ROOT  where models/, runs/, data/ and reports/ live: $GREEKEVAL_PROJECT, else `project_root` in the
                registry file, else the toolkit folder itself
  proj(p)       an absolute path is returned as is, a relative one is taken from PROJECT_ROOT, never from the
                current directory

Paths are built with os.path.join/abspath only, never realpath: lm-eval names each output subfolder after the model
path string it was given, so expanding a symlink would put a second results json next to the existing one.
"""
from __future__ import annotations

import os

import yaml

EVAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS = os.path.join(EVAL_DIR, "lm-eval-adapted/lm-evaluation-harness")
CONFIG_DIR = os.path.join(EVAL_DIR, "config")
REGISTRY = os.path.abspath(os.environ.get("GREEKEVAL_MODELS") or os.path.join(CONFIG_DIR, "models.yaml"))


def _project_root() -> str:
    root = os.environ.get("GREEKEVAL_PROJECT")
    if not root and os.path.exists(REGISTRY):
        with open(REGISTRY, encoding="utf-8") as fh:
            root = (yaml.safe_load(fh) or {}).get("project_root")
    return os.path.abspath(os.path.join(EVAL_DIR, root)) if root else EVAL_DIR   # join keeps an absolute root as is


PROJECT_ROOT = _project_root()


def proj(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
