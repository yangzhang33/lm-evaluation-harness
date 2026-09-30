"""Ground-truth status of the evaluation campaign, read off the filesystem rather than off driver logs.

    python -m greekeval.status              # table + any problems
    python -m greekeval.status --watch 300  # rewrite runs/eval/STATUS.md every 5 minutes

A group counts as done when its output directory holds a `results_*.json`; that is the same test the driver itself
uses to skip work, so this never disagrees with what a re-run would do. Live processes are found by inspecting
/proc and keeping only entries whose executable really is python: matching on the command line alone also matches
the shell that launched the driver (and any shell whose own command line quotes the pattern), which is exactly the
self-match that deadlocked the 2026-09-18 chain script for two hours.
"""
from __future__ import annotations

import argparse
import glob
import os
import time

from .models import MODELS
from .paths import proj
from .run_suite import GROUPS

OUT_ROOT = proj("runs/eval")
GEN_GROUPS = [g[0] for g in GROUPS if g[3]]


def live_drivers() -> list[dict]:
    """Python processes running the driver or lm-eval, with the (model, group) each one is writing to, read out of
    its --output_path. Shells are excluded by construction: only processes whose executable is python are kept."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            exe = os.path.realpath(f"/proc/{pid}/exe")
            if "python" not in os.path.basename(exe):
                continue
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").replace("\0", " ").strip()
        except (OSError, PermissionError):
            continue
        if not any(k in cmd for k in ("lm_eval", "greekeval.run_suite", "greekeval.judge.generate")):
            continue
        parts = cmd.split()
        gpu = next((c for c in parts if c.startswith("cuda:")), "")
        model = group = ""
        if "--output_path" in parts:                     # .../runs/eval/<model>/<group>
            model, group = os.path.abspath(parts[parts.index("--output_path") + 1]).split(os.sep)[-2:]
        else:                                            # a driver / the judge generator: models are on the command line
            model = " ".join(p for p in parts if p in MODELS)
            if "greekeval.judge.generate" in cmd:
                group = "judge-gen"
        out.append(dict(pid=pid, gpu=gpu or "-", model=model, group=group))
    return sorted(out, key=lambda d: d["pid"])


def group_state(model: str, group: str) -> str:
    d = os.path.join(OUT_ROOT, model, group)
    if glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True):
        return "ok"
    if os.path.isdir(d):
        return "partial"     # directory exists but no scores: a crashed or still-running job
    return "-"


def render() -> str:
    cols = [g[0] for g in GROUPS if g[0] != "gen_mmlupro"] + [g + "_raw" for g in GEN_GROUPS]
    live = live_drivers()
    # A column is shown when some model has output in it OR a job is writing to it right now. Hiding a column that is
    # merely still empty is how this table implied, on 2026-09-18, that the raw-completion pass was not running.
    inflight = {(d["model"], d["group"]) for d in live if d["group"]}
    present = [c for c in cols
               if any(group_state(m, c) != "-" for m in MODELS) or any(g == c for _, g in inflight)]
    w = max(len(m) for m in MODELS) + 1
    short = [c.replace("gen_", "g:").replace("mcq_", "m:") for c in present]
    cw = max((len(x) for x in short), default=8) + 2
    head = "model".ljust(w) + "".join(x.rjust(cw) for x in short)
    lines = [head, "-" * len(head)]
    problems = []
    for m in MODELS:
        states = [("running" if (m, c) in inflight else group_state(m, c)) for c in present]
        if all(s == "-" for s in states):
            continue
        label = {"ok": "ok", "partial": "PARTIAL", "running": "RUNNING", "-": "."}
        lines.append(m.ljust(w) + "".join(label[s].rjust(cw) for s in states))
        problems += [f"{m}/{c} has an output directory but no results json"
                     for c, s in zip(present, states) if s == "partial"]
    lines.append("")
    lines.append(f"live python jobs: {len(live)}")
    for d in live:
        where = f"{d['model']}/{d['group']}" if d["group"] else f"driver for {d['model'] or '?'}"
        lines.append(f"  {d['pid']:>8}  {d['gpu']:7} {where}")
    if not live and problems:
        problems.append("nothing is running, so those PARTIAL groups are dead, not in progress")
    if problems:
        lines += ["", "PROBLEMS:"] + [f"  - {p}" for p in problems]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="loop, rewriting runs/eval/STATUS.md")
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "STATUS.md"))
    args = ap.parse_args()
    args.out = proj(args.out)
    while True:
        text = f"# eval status — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n```\n{render()}\n```\n"
        if args.watch:
            tmp = args.out + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(text)
            os.replace(tmp, args.out)          # readers never see a half-written file
        else:
            print(text)
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
