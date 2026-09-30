"""The model registry: which models exist, by key.

The registry is project configuration, not part of the toolkit. It is a YAML file in the project,
`configs/eval/models.yaml` under the project root, or the file named by $GREEKEVAL_MODELS. Start from
`models.example.yaml` next to this package. The columns of every score table follow the order of the file.

Fields of one model:
  path               local model directory; a relative path is taken from the project root (see paths.proj)
  turn_end           the chat template's end-of-turn token, used as the generation stop
  think              how the think block is closed: "enable_thinking" (template switch) or "suffix" (closing tag appended)
  trust_remote_code  passed to lm-eval / transformers
  mcq_batch          "half" = run the likelihood and letter groups at half the group's batch size
  tokenizer          load the tokenizer from this directory instead of `path`
  judge_max_new      new-token budget of the judge generator (default 1024)
  judge              false = the judge generator refuses this model
Top level: `judge_baseline`, the model key the pairwise judge rubric compares against.
"""
from __future__ import annotations

import os

import yaml

from .paths import proj

REGISTRY = proj(os.environ.get("GREEKEVAL_MODELS") or "configs/eval/models.yaml")
if not os.path.exists(REGISTRY):
    raise SystemExit(f"no model registry at {REGISTRY}: copy evaluation/models.example.yaml there and edit it, "
                     "or point $GREEKEVAL_MODELS at your own file")
with open(REGISTRY, encoding="utf-8") as _fh:
    _cfg = yaml.safe_load(_fh) or {}

MODELS: dict[str, dict] = _cfg.get("models") or {}
JUDGE_BASELINE: str | None = _cfg.get("judge_baseline")
