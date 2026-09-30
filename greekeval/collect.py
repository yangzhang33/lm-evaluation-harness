"""Collect every results json under runs/eval into one table: metric x model, with n-shot and stderr.

    python -m greekeval.collect [--root runs/eval] [--csv report.csv] [--md report.md]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

from .models import MODELS
from .paths import proj

MODEL_ORDER = list(MODELS)     # table columns follow the order of the registry
# (task, display name, metric key[, group]). A row may name the driver group it must be read from: greekmmlu is
# scored both 0-shot (mcq_fs0) and 5-shot (mcq_fs5), and without the group the last file glob happened to return
# would silently decide which number is reported. Rows without a group take the task from whichever group ran it.
METRICS = [
    ("greekmmlu",                "GreekMMLU (native) 0-shot", "acc,none", "mcq_fs0"),
    ("greekmmlu_stem",           "  · STEM",                  "acc,none", "mcq_fs0"),
    ("greekmmlu_humanities",     "  · Humanities",            "acc,none", "mcq_fs0"),
    ("greekmmlu_social_sciences","  · Social sciences",       "acc,none", "mcq_fs0"),
    ("greekmmlu_other",          "  · Other",                 "acc,none", "mcq_fs0"),
    ("greekmmlu",                "GreekMMLU (native) 5-shot", "acc,none", "mcq_fs5"),
    # generative re-scoring of the letter-scored MCQ tasks (model writes the letter; raw prompt, same exemplars).
    # Each row is pinned to its group and compared with the log-likelihood row at the same shot count.
    ("greekmmlu_gen",            "GreekMMLU gen 5-shot (writes letter)",   "exact_match,letter", "gen_mmlu_fs5"),
    ("greekmmlu_gen",            "  · parsed",                              "parsed,letter",      "gen_mmlu_fs5"),
    ("greekmmlu_gen_boxed",      "GreekMMLU gen 0-shot (boxed)",            "exact_match,boxed",  "gen_mmlu_fs0"),
    ("greekmmlu_gen_boxed",      "  · parsed",                              "parsed,boxed",       "gen_mmlu_fs0"),
    # the same letter-writing tasks WITH each model's chat template (the `_chat` lines of the plan, output <group>_chat/)
    ("greekmmlu_gen",            "GreekMMLU gen 5-shot, chat template",     "exact_match,letter", "gen_mmlu_fs5_chat"),
    ("greekmmlu_gen",            "  · parsed",                              "parsed,letter",      "gen_mmlu_fs5_chat"),
    ("greekmmlu_gen_boxed",      "GreekMMLU gen 0-shot, chat template",     "exact_match,boxed",  "gen_mmlu_fs0_chat"),
    ("greekmmlu_gen_boxed",      "  · parsed",                              "parsed,boxed",       "gen_mmlu_fs0_chat"),
    ("ilspgreekasep",            "ASEP exams (native)",       "acc,none"),
    ("ilspgreekmedicalmcqa",     "Medical MCQA (native)",     "acc,none"),
    ("belebele_ell_Grek",        "Belebele el (human MT)",    "acc,none"),
    ("belebele_ell_Grek_gen",    "Belebele el gen 5-shot (writes letter)", "exact_match,letter", "gen_bele_fs5"),
    ("belebele_ell_Grek_gen",    "Belebele el gen 5-shot, chat template",  "exact_match,letter", "gen_bele_fs5_chat"),
    ("ilspgreekmmlu",            "MMLU el (MT) · first 40/subject", "acc,none"),
    ("ilspgreekmmlu_gen",        "MMLU el gen 5-shot (writes letter)",     "exact_match,letter", "gen_ilsp_fs5"),
    # the letter run is full (14,042) but the likelihood row above is the first 40/subject, and the first items are
    # not a random sample, so the fair side-by-side is the letter run re-scored on those same items, from its samples
    ("ilspgreekmmlu_gen",        "  · same first 40/subject as MMLU el",   "exact_match,letter", "gen_ilsp_fs5", 40),
    ("ilspgreekmmlu_gen",        "MMLU el gen 5-shot, chat template",      "exact_match,letter", "gen_ilsp_fs5_chat"),
    ("ilspgreekhellaswag",       "HellaSwag el (MT) · first 2,000", "acc_norm,none"),
    ("ilspgreekarc_easy",        "ARC-Easy el (MT)",          "acc_norm,none"),
    ("ilspgreekarc_challenge",   "ARC-Challenge el (MT)",     "acc_norm,none"),
    ("ilspgreekwinogrande",      "WinoGrande el (MT, filtered)", "acc,none"),
    ("ilspgreektruthfulqa_mc1",  "TruthfulQA el mc1",         "acc,none"),
    ("ilspgreektruthfulqa_mc2",  "TruthfulQA el mc2",         "acc,none"),
    ("mmlu",                     "MMLU English (guard) · first 40/subject", "acc,none"),
    ("mmlu_generative",          "MMLU English gen 5-shot (writes letter) · first 40/subject", "exact_match,get_response", "gen_mmlu_en_fs5"),
    ("mmlu_generative",          "MMLU English gen 5-shot, chat template · first 40/subject", "exact_match,get_response", "gen_mmlu_en_fs5_chat"),
    ("ilspgreekifeval",          "IFEval el · prompt strict", "prompt_level_strict_acc,none"),
    ("ilspgreekifeval",          "IFEval el · inst strict",   "inst_level_strict_acc,none"),
    ("ilspgreekifeval",          "IFEval el · inst loose",    "inst_level_loose_acc,none"),
    ("ifeval",                   "IFEval English · inst strict", "inst_level_strict_acc,none"),
    ("ilspgreekmgsm",            "MGSM el (flexible)",        "exact_match,flexible-extract", "gen_fs8b"),  # 2048-token cap; gen_fs8 (256) is the retired artifact
    ("ilspgreekcivicsqa",        "Civics QA el · BLEU",       "bleu,none"),
    ("ilspgreekflores_en_el",    "FLORES en→el · chrF",       "chrf,none"),
    ("ilspgreekflores_el_en",    "FLORES el→en · chrF",       "chrf,none"),
    ("ilspgreektruthfulqa_gen",  "TruthfulQA el gen · bleu_acc", "bleu_acc,none"),
]


def load(root: str, protocol: str = "chat"):
    """Results are keyed by (model, task|metric), so the two protocols MUST NOT be read together: a `<group>_raw`
    run scores the same task names as its chat-template counterpart and would silently overwrite them in whichever
    order glob happened to return the files. `chat` reads everything except the _raw directories, `raw` only those."""
    vals: dict[tuple[str, str], float] = {}
    shots: dict[str, str] = {}
    for f in glob.glob(os.path.join(root, "*", "*", "*", "results_*.json")):
        model, group = f[len(root) + 1:].split(os.sep)[:2]
        if group.endswith("_raw") != (protocol == "raw"):
            continue
        r = json.load(open(f))
        ns = r.get("n-shot", {}) or {}
        for task, m in r["results"].items():
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    vals[(model, f"{task}|{k}")] = v            # group-agnostic (last wins) ...
                    vals[(model, f"{group}/{task}|{k}")] = v    # ... and group-qualified, for rows that pin one
            if task in ns:
                shots[task] = str(ns[task]); shots[f"{group}/{task}"] = str(ns[task])
    return vals, shots


def first_n_score(root: str, model: str, group: str, task: str, metric: str, n: int) -> float | None:
    """Mean of `metric` over the first n docs of every subtask of `task`, read from the logged samples: the same
    items an lm-eval `--limit n` run scores."""
    field, flt = metric.split(",")
    hits = []
    for f in glob.glob(os.path.join(root, model, group, "*", f"samples_{task}_*.jsonl")):
        hits += [r[field] for r in map(json.loads, open(f)) if r["doc_id"] < n and r.get("filter", flt) == flt]
    return sum(hits) / len(hits) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/eval")
    ap.add_argument("--protocol", choices=["chat", "raw"], default="chat",
                    help="chat = the campaign protocol (chat template for generation); raw = the base-model "
                         "protocol in <group>_raw/ (no template, each task's own stop rules). Never mix them.")
    ap.add_argument("--csv"); ap.add_argument("--md")
    args = ap.parse_args()
    args.root = proj(args.root)
    vals, shots = load(args.root, args.protocol)
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in vals)]

    rows = []
    for task, label, metric, *grp in METRICS:
        if args.protocol != "chat":      # the raw protocol has one group per task; pins name chat groups only
            grp = []
        key = (f"{grp[0]}/" if grp else "") + f"{task}|{metric}"
        if not any((m, key) in vals for m in models):
            continue
        cells = []
        for m in models:
            v = vals.get((m, key))
            if len(grp) > 1 and v is not None:          # (group, first_n): re-score the run on its first n items
                v = first_n_score(args.root, m, grp[0], task, metric, grp[1])
            pre = f"{grp[0]}/" if grp else ""
            se = vals.get((m, f"{pre}{task}|{metric.split(',')[0]}_stderr,{metric.split(',')[1]}"))
            cells.append((v, se))
        rows.append((label, shots.get(f"{pre}{task}", shots.get(task, "-")), cells))

    w = max(len(r[0]) for r in rows) + 1
    head = f"{'metric':<{w}} {'shot':>4} " + " ".join(f"{m:>10}" for m in models)
    print(head); print("-" * len(head))
    for label, shot, cells in rows:
        line = f"{label:<{w}} {shot:>4} " + " ".join(
            ("     -    " if v is None else f"{v:>10.4f}") for v, _ in cells)
        print(line)

    if args.csv:
        with open(proj(args.csv), "w", newline="") as fh:
            cw = csv.writer(fh); cw.writerow(["metric", "n_shot", *models])
            for label, shot, cells in rows:
                cw.writerow([label.strip(), shot, *["" if v is None else round(v, 4) for v, _ in cells]])
        print(f"\ncsv -> {args.csv}")
    if args.md:
        with open(proj(args.md), "w") as fh:
            fh.write("| metric | shot | " + " | ".join(models) + " |\n")
            fh.write("|---|---|" + "---|" * len(models) + "\n")
            for label, shot, cells in rows:
                fh.write(f"| {label.strip()} | {shot} | " +
                         " | ".join("-" if v is None else f"{v:.4f}" for v, _ in cells) + " |\n")
        print(f"md -> {args.md}")


if __name__ == "__main__":
    main()
