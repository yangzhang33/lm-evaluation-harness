"""The model registry: which models exist, by key.

The registry is local configuration: `config/models.yaml` next to this package (git-ignored, so local paths and
model names never leave the machine), or the file named by $GREEKEVAL_MODELS. Start from `config/models.example.yaml`.
The columns of every score table follow the order of the file.

Fields of one model:
  path               local model directory; a relative path is taken from the project root (see paths.proj)
  turn_end           the chat template's end-of-turn token, used as the generation stop
  think              how the think block is closed: "enable_thinking" (template switch) or "suffix" (closing tag appended)
  trust_remote_code  passed to lm-eval / transformers
  mcq_batch          "half" = run the likelihood and letter groups at half the group's batch size
  tokenizer          load the tokenizer from this directory instead of `path`
  judge_max_new      new-token budget of the judge generator (default 1024)
  judge              false = the judge generator refuses this model
Top level: `project_root`, where models/, runs/, data/ and reports/ live (default: the toolkit folder), and
`judge_baseline`, the model key the pairwise judge rubric compares against.
"""
from __future__ import annotations

import os

import yaml

from .paths import REGISTRY

if not os.path.exists(REGISTRY):
    raise SystemExit(f"no model registry at {REGISTRY}: copy config/models.example.yaml to config/models.yaml and "
                     "list your models, or point $GREEKEVAL_MODELS at your own file")
with open(REGISTRY, encoding="utf-8") as _fh:
    _cfg = yaml.safe_load(_fh) or {}

MODELS: dict[str, dict] = _cfg.get("models") or {}
JUDGE_BASELINE: str | None = _cfg.get("judge_baseline")
