"""Ground-truth status of the evaluation campaign, read off the filesystem rather than off driver logs.

    python -m greekeval.status              # table + any problems
    python -m greekeval.status --watch 300  # rewrite runs/eval/STATUS.md every 5 minutes

A run counts as done when its output directory holds a `results_*.json`; that is the same test the driver itself
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
from .paths import CONFIG_DIR, proj
from .plan import known_runs, load_plan


def live_drivers() -> list[dict]:
    """Python processes running the driver or lm-eval, with the (model, run) each one is writing to, read out of
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
        if "--output_path" in parts:                     # .../<root>/<model>/<run>
            model, group = os.path.abspath(parts[parts.index("--output_path") + 1]).split(os.sep)[-2:]
        else:                                            # a driver / the judge generator: models are on the command line
            model = " ".join(p for p in parts if p in MODELS)
            if "greekeval.judge.generate" in cmd:
                group = "judge-gen"
        out.append(dict(pid=pid, gpu=gpu or "-", model=model, group=group))
    return sorted(out, key=lambda d: d["pid"])


def run_state(root: str, model: str, group: str) -> str:
    d = os.path.join(root, model, group)
    files = glob.glob(os.path.join(d, "**", "results_*.json"), recursive=True)
    if len(files) > 1:
        return "dup"         # two results files: the collector would mix them (see "Output format", rule 2)
    if files:
        return "ok"
    if os.path.isdir(d):
        return "partial"     # directory exists but no scores: a crashed or still-running job
    return "-"


def render(plan: list[dict], runs_root: str, known: set[tuple[str, str]] = frozenset()) -> str:
    lines, problems, live = [], [], live_drivers()
    inflight = {(d["model"], d["group"]) for d in live if d["group"]}
    for root in ("eval", "eval_think"):
        rdir = os.path.join(runs_root, root)
        # the layout is <runs>/<root>/<model key>/<plan line name>/: anything else at those two levels is a misplaced
        # or misnamed run that the collector would either miss or file under the wrong name
        if os.path.isdir(rdir):
            problems += [f"{root}/{d} is not a model key of the registry" for d in sorted(os.listdir(rdir))
                         if os.path.isdir(os.path.join(rdir, d)) and d not in MODELS and d != "logs"]
        cols = [r["name"] for r in plan if r["root"] == root]
        extra = sorted({g for m in MODELS for g in (os.listdir(os.path.join(rdir, m)) if os.path.isdir(os.path.join(rdir, m)) else [])
                        if g not in cols})
        problems += [f"{root}/{m}/{g} is not a line of the plan, commented out or not (misnamed run?)"
                     for g in extra for m in MODELS if os.path.isdir(os.path.join(rdir, m, g)) and (root, g) not in known]
        cols += extra                           # folders of commented-out lines (e.g. retired runs) still show
        # A column is shown when some model has output in it OR a job is writing to it right now. Hiding a column that
        # is merely still empty is how this table implied, on 2026-09-18, that the raw-completion pass was not running.
        present = [c for c in cols if any(run_state(rdir, m, c) != "-" for m in MODELS) or any(g == c for _, g in inflight)]
        if not present:
            continue
        w = max(len(m) for m in MODELS) + 1
        short = [c.replace("gen_", "g:").replace("mcq_", "m:") for c in present]
        cw = max((len(x) for x in short), default=8) + 2
        head = f"{root}/".ljust(w) + "".join(x.rjust(cw) for x in short)
        lines += [head, "-" * len(head)]
        for m in MODELS:
            states = [("running" if (m, c) in inflight else run_state(rdir, m, c)) for c in present]
            if all(s == "-" for s in states):
                continue
            label = {"ok": "ok", "partial": "PARTIAL", "running": "RUNNING", "dup": "DUP", "-": "."}
            lines.append(m.ljust(w) + "".join(label[s].rjust(cw) for s in states))
            problems += [f"{root}/{m}/{c} has an output directory but no results json" for c, s in zip(present, states) if s == "partial"]
            problems += [f"{root}/{m}/{c} holds more than one results json: delete the folder and re-run it"
                         for c, s in zip(present, states) if s == "dup"]
        lines.append("")
    lines.append(f"live python jobs: {len(live)}")
    for d in live:
        where = f"{d['model']}/{d['group']}" if d["group"] else f"driver for {d['model'] or '?'}"
        lines.append(f"  {d['pid']:>8}  {d['gpu']:7} {where}")
    if not live and any("no results json" in p for p in problems):
        problems.append("nothing is running, so those PARTIAL runs are dead, not in progress")
    if problems:
        lines += ["", "PROBLEMS:"] + [f"  - {p}" for p in problems]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(CONFIG_DIR, "plan.yaml"))
    ap.add_argument("--runs", default="runs", help="folder that holds eval/ and eval_think/ (default: runs)")
    ap.add_argument("--watch", type=int, metavar="SECONDS", help="loop, rewriting <runs>/eval/STATUS.md")
    ap.add_argument("--out", help="where --watch writes (default: <runs>/eval/STATUS.md)")
    args = ap.parse_args()
    plan, runs_root, known = load_plan(proj(args.plan)), proj(args.runs), known_runs(proj(args.plan))
    out = proj(args.out) if args.out else os.path.join(runs_root, "eval", "STATUS.md")
    while True:
        text = f"# eval status — {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n```\n{render(plan, runs_root, known)}\n```\n"
        if args.watch:
            tmp = out + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(text)
            os.replace(tmp, out)          # readers never see a half-written file
        else:
            print(text)
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
