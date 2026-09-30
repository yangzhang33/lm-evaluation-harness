"""Every evaluation number in one long-format table: reports/eval_results/all_scores.csv.

    python -m greekeval.export_scores [--runs runs] [--out reports/eval_results/all_scores.csv]

Reads every lm-eval results json under <runs>/eval (the campaign) and <runs>/eval_think (think-mode diagnostics) and
writes one row per (model, run, task, metric): value, stderr, shots, sample count, item cap, generation cap, and how
the task was scored. The per-run jsons stay in runs/ (git-ignored); this file is the version-controlled record.

Columns
  model, group        registry key and run name (the folder names under <runs>/eval[_think]/<model>/)
  task                lm-eval task or subtask (e.g. greekmmlu, greekmmlu_stem, greekmmlu_law)
  metric, value, stderr
  scoring             loglik = rank the options' likelihood; generate = the model writes the answer
  template            raw = plain few-shot prompt; chat = the model's chat template
  think               closed / open = state of the think block; none = the model has no think block; n/a = no template
  n_shot, n_samples, limit (per-subtask cap, empty = full set)
  source              runs/eval or runs/eval_think
  note                known caveats, e.g. a retired run
  max_gen_toks        the generation cap the run was made with (empty = the task's own default)
Everything but `note` is read from the results json itself, so a run made without the driver is described the same way.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

from .paths import proj

OUT = "reports/eval_results/all_scores.csv"
NOTES = {"gen_fs8": "retired: lm-eval's 256-token default cut answers; use gen_fs8b"}
FIELDS = ["model", "group", "task", "metric", "value", "stderr", "scoring", "template", "think",
          "n_shot", "n_samples", "limit", "source", "note", "max_gen_toks"]


def protocol(r: dict) -> tuple[str, str, str, str]:
    """(scoring, template, think, max_gen_toks) of one results json."""
    types = {c.get("output_type") for c in (r.get("configs") or {}).values()}
    scoring = "loglik" if types and types <= {"multiple_choice", "loglikelihood", "loglikelihood_rolling"} else "generate"
    cfg = r.get("config") or {}
    margs = cfg.get("model_args") if isinstance(cfg.get("model_args"), dict) else {}
    gk = cfg.get("gen_kwargs") if isinstance(cfg.get("gen_kwargs"), dict) else {}
    cap = "" if gk.get("max_gen_toks") is None else int(gk["max_gen_toks"])
    if scoring == "loglik" or not r.get("chat_template"):
        return scoring, "raw", "n/a", cap
    if margs.get("enable_thinking") is True or "think_end_token" in margs:
        think = "open"
    elif margs.get("enable_thinking") is False or "chat_template_suffix" in margs:
        think = "closed"
    else:
        think = "none"
    return scoring, "chat", think, cap


def latest_results(root: str) -> dict[tuple[str, str], str]:
    out = {}
    for f in glob.glob(f"{root}/*/*/**/results_*.json", recursive=True):
        model, group = os.path.relpath(f, root).split(os.sep)[:2]
        if (model, group) not in out or f > out[(model, group)]:
            out[(model, group)] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="folder that holds eval/ and eval_think/ (default: runs)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rows = []
    for root in ("eval", "eval_think"):
        source = f"{args.runs}/{root}"
        for (model, group), f in sorted(latest_results(proj(source)).items()):
            r = json.load(open(f))
            shots, ns, limit = r.get("n-shot", {}) or {}, r.get("n-samples", {}) or {}, (r.get("config") or {}).get("limit")
            scoring, template, think, cap = protocol(r)
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
                        "limit": "" if limit is None else int(limit), "source": source, "note": NOTES.get(group, ""),
                        "max_gen_toks": cap,
                    })
    out = proj(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    models = sorted({r["model"] for r in rows}); groups = sorted({(r["model"], r["group"], r["source"]) for r in rows})
    print(f"{args.out}: {len(rows)} rows, {len(models)} models, {len(groups)} (model, group) runs")


if __name__ == "__main__":
    main()
