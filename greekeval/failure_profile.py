"""Generation failure profile: rule-based diagnostics over lm-eval sample dumps.

Multiple-choice benchmarks score by ranking the log-likelihood of the given options, so the
model never generates a token: GreekMMLU cannot see whether a model stops, answers in Greek,
or loops. This reads the `samples_<task>_*.jsonl` files that `lm-eval --log_samples` writes
and measures those properties on generations we already produced — no extra prompts, no
judge, no new benchmark to defend.

    python -m greekeval.failure_profile \
        --runs base=/tmp/eval/base sft=/tmp/eval/sft \
        --tokenizer <HF id or model dir> [--json profile.json]

Each run path is an lm-eval `--output_path` directory (or a single samples jsonl). Run base
and SFT models through the *identical* generation path, or the difference you measure is the
prompt template, not the training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import Counter

from .langid import GreekLangID
from .log import get_logger
from .paths import proj

log = get_logger("greekeval.failure_profile")

# Language each task is supposed to answer in. Greek unless the task asks otherwise: the suite also carries the
# English forgetting guards and the Greek->English translation direction, where a Greek answer is the failure.
EXPECTED_LANG = {
    "ilspgreekflores_el_en": "eng",   # translate INTO English
    "ifeval": "eng",                  # upstream English IFEval (guard)
    "mmlu": "eng",
    "gsm8k": "eng",
    "truthfulqa_gen": "eng",
}
DEFAULT_LANG = "ell"

SHORT_CHARS = 20  # below this a reply is not an attempt
NGRAM = 5
# Share of duplicate 5-grams. Calibrated on real IFEval generations: healthy replies (lists
# and templates included) sit far below the threshold, degenerate ones far above it.
REPEAT_RATIO = 0.30


def find_samples(path: str) -> dict[str, str]:
    """Newest samples file per task under an lm-eval output dir."""
    if path.endswith(".jsonl"):
        files = [path]
    else:
        files = glob.glob(os.path.join(path, "**", "samples_*.jsonl"), recursive=True)
    newest: dict[str, str] = {}
    for f in sorted(files):
        task = os.path.basename(f)[len("samples_"):].rsplit("_", 1)[0]
        newest[task] = f  # sorted by name, so the last timestamp wins
    return newest


def load_tokenizer(spec: str):
    from tokenizers import Tokenizer

    local = os.path.join(proj(spec), "tokenizer.json")
    if os.path.exists(local):
        return Tokenizer.from_file(local)
    from huggingface_hub import hf_hub_download

    return Tokenizer.from_file(hf_hub_download(spec, "tokenizer.json"))


def response_of(row: dict) -> str:
    r = row.get("resps") or [[""]]
    r = r[0]
    return (r[0] if isinstance(r, list) else r) or ""


def repetition_ratio(text: str) -> float:
    """Share of 5-grams that are duplicates. Robust where a raw repeat count is not:
    a bulleted list or a form template legitimately repeats a 5-gram a few times, and
    IFEval even has a constraint that asks for the prompt to be repeated."""
    words = text.split()
    if len(words) < NGRAM * 3:
        return 0.0
    grams = [tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)]
    return 1 - len(set(grams)) / len(grams)


def mojibake(text: str) -> bool:
    if "\ufffd" in text:
        return True
    return len(text) > 50 and sum(c.isalpha() for c in text) / len(text) < 0.3


def profile_task(rows: list[dict], langid: GreekLangID, tok=None) -> dict:
    gen_kwargs = (rows[0].get("arguments", {}).get("gen_args_0", {}) or {}).get("arg_1") or {}
    if not isinstance(gen_kwargs, dict):  # loglikelihood task: arg_1 is the continuation string, nothing to profile
        return None
    # Termination is only observable when nothing else stops the generation for the model.
    measurable_termination = gen_kwargs.get("until") == [] and tok is not None
    cap = gen_kwargs.get("max_gen_toks")
    task_name = rows[0].get("_task", "")
    expected = EXPECTED_LANG.get(task_name, DEFAULT_LANG)

    n = len(rows)
    empty = short = wrong_lang = looping = broken = truncated = 0
    lengths: list[int] = []
    for row in rows:
        text = response_of(row)
        stripped = text.strip()
        if not stripped:
            empty += 1
            continue
        lengths.append(len(stripped))
        if len(stripped) < SHORT_CHARS:
            short += 1
        _, label = langid.is_greek(stripped)
        if not label.split(":")[0].startswith(expected):
            wrong_lang += 1
        if repetition_ratio(stripped) >= REPEAT_RATIO:
            looping += 1
        if mojibake(stripped):
            broken += 1
        if measurable_termination and len(tok.encode(text, add_special_tokens=False).ids) >= cap:
            truncated += 1

    # Whole-population rates for empty and termination; quality rates over the replies the
    # model actually produced, so that a model which mostly stays silent cannot look clean.
    answered = n - empty
    per_answer = (lambda c: c / answered) if answered else (lambda c: None)
    out = {
        "n": n,
        "answered": answered,
        "expected_lang": expected,
        "empty_rate": empty / n,
        "short_rate": per_answer(short),
        "wrong_lang_rate": per_answer(wrong_lang),
        "loop_rate": per_answer(looping),
        "mojibake_rate": per_answer(broken),
        "median_chars": statistics.median(lengths) if lengths else 0,
    }
    # A reply that neither came back empty nor ran into the token cap ended on its own.
    out["termination_rate"] = (n - empty - truncated) / n if measurable_termination else None
    strict = [row["inst_level_strict_acc"] for row in rows if "inst_level_strict_acc" in row]
    if strict:
        flat = [bool(v) for sub in strict for v in (sub if isinstance(sub, list) else [sub])]
        out["format_pass_rate"] = sum(flat) / len(flat)
    return out


ROWS = [
    ("n", "samples", "{:.0f}"),
    ("termination_rate", "terminated", "{:.1%}"),
    ("empty_rate", "empty", "{:.1%}"),
    ("answered", "answered", "{:.0f}"),
    ("wrong_lang_rate", "wrong language*", "{:.1%}"),
    ("loop_rate", "repetition loop*", "{:.1%}"),
    ("mojibake_rate", "mojibake*", "{:.1%}"),
    ("short_rate", "under 20 chars*", "{:.1%}"),
    ("median_chars", "median chars*", "{:.0f}"),
    ("format_pass_rate", "format constraints", "{:.1%}"),
]


def render(profiles: dict[str, dict[str, dict]]) -> str:
    runs = list(profiles)
    lines = []
    for task in sorted({t for p in profiles.values() for t in p}):
        present = [r for r in runs if task in profiles[r]]
        width = max(14, *(len(r) for r in present))
        lines.append(f"\n{task}  (expects {profiles[present[0]][task]['expected_lang']})")
        lines.append("  " + "metric".ljust(20) + "".join(r.rjust(width + 2) for r in present))
        for key, label, fmt in ROWS:
            vals = [profiles[r][task].get(key) for r in present]
            if all(v is None for v in vals):
                continue
            cells = ["n/a".rjust(width + 2) if v is None else fmt.format(v).rjust(width + 2) for v in vals]
            lines.append("  " + label.ljust(20) + "".join(cells))
    lines.append("\n  * rates marked with an asterisk are over answered samples, not all samples")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, metavar="LABEL=PATH",
                    help="lm-eval --output_path dirs, e.g. base=/tmp/eval/base sft=/tmp/eval/sft")
    ap.add_argument("--tokenizer", help="HF id or local dir; needed for the termination rate")
    ap.add_argument("--tasks", nargs="*", help="limit to these tasks")
    ap.add_argument("--json", help="write the raw numbers here")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer) if args.tokenizer else None
    if tok is None:
        log.warning("no --tokenizer: termination rate cannot be measured")
    langid = GreekLangID()

    profiles: dict[str, dict[str, dict]] = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            label, path = os.path.basename(path or spec.rstrip("/")), spec
        profiles[label] = {}
        for task, f in find_samples(proj(path)).items():
            if args.tasks and task not in args.tasks:
                continue
            rows = [json.loads(line) for line in open(f, encoding="utf-8")]
            if not rows:
                continue
            for row in rows:
                row["_task"] = task
            prof = profile_task(rows, langid, tok)
            if prof is None:
                continue
            profiles[label][task] = prof
            log.info("%s / %s: %d samples from %s", label, task, len(rows), os.path.basename(f))

    print(render(profiles))
    if args.json:
        with open(proj(args.json), "w", encoding="utf-8") as fh:
            json.dump(profiles, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
