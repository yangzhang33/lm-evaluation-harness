"""Every evaluation number in one long-format table: reports/eval_results/all_scores.csv.

    python -m greekeval.export_scores

Reads every lm-eval results json under runs/eval (the campaign) and runs/eval_think (think-mode diagnostics) and writes
one row per (model, group, task, metric): value, stderr, shots, sample count, item cap, and how the task was scored.
The per-run jsons stay in runs/ (git-ignored); this file is the version-controlled record of all scores.

Columns
  model, group        run_suite model key and driver group (the folder name under runs/eval[_think]/<model>/)
  task                lm-eval task or subtask (e.g. greekmmlu, greekmmlu_stem, greekmmlu_law)
  metric, value, stderr
  scoring             loglik = rank the options' likelihood; generate = the model writes the answer
  template            raw = plain few-shot prompt; chat = the model's chat template
  think               closed / open = state of the think block; none = model has no think block; n/a = no template
  n_shot, n_samples, limit (per-subtask cap, empty = full set)
  source              runs/eval or runs/eval_think
  note                known caveats, e.g. a retired group
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

from .models import MODELS
from .paths import proj
from .run_suite import LETTER_GROUPS

OUT = "reports/eval_results/all_scores.csv"
NOTES = {"gen_fs8": "retired: lm-eval's 256-token default cut answers; use gen_fs8b"}
FIELDS = ["model", "group", "task", "metric", "value", "stderr", "scoring", "template", "think",
          "n_shot", "n_samples", "limit", "source", "note"]


def protocol(model: str, group: str, root: str) -> tuple[str, str, str]:
    has_think = bool(MODELS.get(model, {}).get("think"))
    base = group.removesuffix("_raw").removesuffix("_chat_closed").removesuffix("_chat")
    if group.startswith("mcq"):
        return "loglik", "raw", "n/a"
    if group.endswith("_raw") or (base in LETTER_GROUPS and not group.endswith(("_chat", "_chat_closed"))):
        return "generate", "raw", "n/a"
    think = ("open" if root.endswith("eval_think") and not group.endswith("_closed") else "closed") if has_think else "none"
    return "generate", "chat", think


def latest_results(root: str) -> dict[tuple[str, str], str]:
    out = {}
    for f in glob.glob(f"{root}/*/*/**/results_*.json", recursive=True):
        model, group = os.path.relpath(f, root).split(os.sep)[:2]
        if (model, group) not in out or f > out[(model, group)]:
            out[(model, group)] = f
    return out


def main() -> None:
    rows = []
    for root in ("runs/eval", "runs/eval_think"):
        for (model, group), f in sorted(latest_results(proj(root)).items()):
            r = json.load(open(f))
            shots, ns, limit = r.get("n-shot", {}) or {}, r.get("n-samples", {}) or {}, (r.get("config") or {}).get("limit")
            scoring, template, think = protocol(model, group, root)
            for task, metrics in r["results"].items():
                for key, val in metrics.items():
                    if not isinstance(val, (int, float)) or "_stderr" in key or key == "sample_count":
                        continue
                    name, _, flt = key.partition(",")
                    se = metrics.get(f"{name}_stderr,{flt}")
                    n = ns.get(task, {})
                    rows.append({
                        "model": model, "group": group, "task": task, "metric": key,
                        "value": round(float(val), 6),
                        "stderr": "" if not isinstance(se, (int, float)) or math.isnan(se) else round(float(se), 6),
                        "scoring": scoring, "template": template, "think": think,
                        "n_shot": shots.get(task, ""), "n_samples": n.get("effective", "") if isinstance(n, dict) else "",
                        "limit": "" if limit is None else int(limit), "source": root, "note": NOTES.get(group, ""),
                    })
    os.makedirs(os.path.dirname(proj(OUT)), exist_ok=True)
    with open(proj(OUT), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    models = sorted({r["model"] for r in rows}); groups = sorted({(r["model"], r["group"], r["source"]) for r in rows})
    print(f"{OUT}: {len(rows)} rows, {len(models)} models, {len(groups)} (model, group) runs")


if __name__ == "__main__":
    main()
