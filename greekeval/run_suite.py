"""Drive a Greek eval campaign: every model x every run in the plan, resumable.

    python -m greekeval.run_suite --list --models <key>                        # what would run, what is done
    python -m greekeval.run_suite --gpu 0 --models <key> [<key> ...]           # run the plan
    python -m greekeval.run_suite --gpu 0 --models <key> --limit 1 --runs runs/smoke   # smoke test, 1 item per task

What runs is decided by the plan file, config/plan.yaml next to this package: one line per run; comment out what
you do not want. --groups narrows it to the named runs. Models come from the
project's registry (models.py).
Results land in <runs>/<root>/<model>/<name>/ (root = eval, or eval_think for think-on runs); a run whose folder
already holds a results json is skipped, so a campaign can be resumed by running the same command again.
Relative paths (model paths in the registry, --runs) are taken from the project root, not from the current
directory: see paths.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from .models import MODELS
from .paths import CONFIG_DIR, HARNESS, proj
from .plan import load_plan

# closing tag of the think block, per template style (the registry's `think` field)
THINK_END = {"enable_thinking": "</think>", "suffix": "</ifm|think>"}


def done(out_dir: str) -> bool:
    return any(f.startswith("results_") for _, _, fs in os.walk(out_dir) for f in fs) if os.path.isdir(out_dir) else False


def resolve_batch(model: str, run: dict) -> str:
    """The plan's batch sizes fit a 4B model on a 48GB card; bigger models OOM at them on the long few-shot contexts of
    the multiple-choice forms (A and B). The registry's `mcq_batch` policy applies there: `half` halves the plan's
    size, `auto` hands sizing to lm-eval (`auto:4`, re-detected four times per run). `auto` is safe only when context
    lengths are uniform: it sizes the batch on the contexts it sees first and then OOMs on the long tail. Batch size
    does not change a likelihood score, only the time and memory to compute it."""
    policy = MODELS[model].get("mcq_batch")
    if not policy or run["form"] not in ("A", "B"):
        return run["batch"]
    return "auto:4" if policy == "auto" else str(max(1, int(run["batch"]) // 2))


def build_cmd(model: str, run: dict, out_dir: str, gpu: int, limit: int | None, dp: int = 1) -> list[str]:
    m = MODELS[model]
    chat, think = run["template"] == "chat", run["think"] == "on"
    margs = {"pretrained": os.path.abspath(proj(m["path"])), "dtype": "bfloat16"}
    if m.get("revision"):
        margs["revision"] = m["revision"]
    if m.get("trust_remote_code"):
        margs["trust_remote_code"] = True
    if m.get("tokenizer"):
        margs["tokenizer"] = os.path.abspath(proj(m["tokenizer"]))
    if chat:
        # Think off (the default): the generation prompt carries an empty, CLOSED think block, the prefix a
        # non-thinking fine-tune is trained with. Two template styles, named by the registry's `think` field:
        #   enable_thinking (Qwen): the template has a switch; enable_thinking=False emits "<think>\n\n</think>\n\n".
        #   suffix (K2):            the template always opens the block, so the closing tag is appended verbatim.
        # Think on: the block is left OPEN so the model reasons first, and lm-eval strips everything up to
        # think_end_token before scoring. A model without a `think` field has no block to open or close.
        if m.get("think") == "enable_thinking":
            margs["enable_thinking"] = think
            if think:
                margs["think_end_token"] = THINK_END["enable_thinking"]
        elif m.get("think") == "suffix":
            if think:
                margs["think_end_token"] = THINK_END["suffix"]
            else:
                margs["chat_template_suffix"] = THINK_END["suffix"]
    # --dp N: one lm-eval run data-parallel over GPUs 0..N-1 (accelerate splits the docs across ranks and rank 0 writes
    # a single results file and the gathered samples), so the output folder is the same as a one-GPU run
    launcher = [sys.executable, "-m", "lm_eval"] if dp == 1 else \
        [os.path.join(os.path.dirname(sys.executable), "accelerate"), "launch", "--multi_gpu", "--num_processes", str(dp),
         "--num_machines", "1", "--mixed_precision", "no", "--dynamo_backend", "no", "--main_process_port", "29533",
         "-m", "lm_eval"]
    cmd = launcher + ["--model", "hf", "--model_args", json.dumps(margs), "--tasks", run["tasks"],
           "--num_fewshot", str(run["shots"]), "--batch_size", resolve_batch(model, run)] + \
          (["--device", f"cuda:{gpu}"] if dp == 1 else []) + \
          ["--output_path", os.path.abspath(out_dir), "--log_samples", "--seed", "1234"]
    if chat:
        cmd.append("--apply_chat_template")
        # stop at the chat turn end: a base snapshot may ship no generation_config and a tokenizer eos that is not the
        # turn end, and task-level `until` lists such as MGSM's ["\n\n", "\n"] would cut a multi-line chat answer after
        # its first line. lm-eval replaces the task's `until` with this one.
        gk: dict = {"until": [m["turn_end"]]}
        if run["max_gen_toks"]:
            gk["max_gen_toks"] = run["max_gen_toks"]
        cmd += ["--gen_kwargs", json.dumps(gk)]
        if run["shots"] > 0:
            # this fork auto-enables multi-turn few-shot with a chat template. A `suffix`-style template rejects
            # assistant messages without a `reasoning_content` field, which is what multi-turn few-shot builds, so
            # such a model gets the exemplars concatenated inside one user turn instead.
            cmd += ["--fewshot_as_multiturn", "false" if m.get("think") == "suffix" else "true"]
    elif run["max_gen_toks"]:
        # raw completion keeps the task's own `until`; lm-eval merges the cap into it
        cmd += ["--gen_kwargs", json.dumps({"max_gen_toks": run["max_gen_toks"]})]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def main() -> None:
    # less fragmentation near the memory limit; PyTorch's own OOM message recommends it
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dp", type=int, default=1, help="data-parallel over GPUs 0..N-1 in one run (ignores --gpu)")
    ap.add_argument("--models", nargs="+", default=list(MODELS), help="registry keys (default: every model)")
    ap.add_argument("--plan", default=os.path.join(CONFIG_DIR, "plan.yaml"), help="the plan file (default: config/plan.yaml)")
    ap.add_argument("--groups", nargs="+", metavar="NAME",
                    help="only the plan runs with these names (default: every line that is not commented out)")
    ap.add_argument("--runs", default="runs", help="folder that holds eval/ and eval_think/ (default: runs)")
    ap.add_argument("--limit", type=int, help="cap items per task, for smoke tests; needs a --runs other than runs")
    ap.add_argument("--list", action="store_true", help="show the plan for these models and exit")
    args = ap.parse_args()

    plan = load_plan(proj(args.plan))
    if args.groups:
        unknown = sorted(set(args.groups) - {r["name"] for r in plan})
        if unknown:
            raise SystemExit(f"not in the plan (or commented out): {' '.join(unknown)}")
        plan = [r for r in plan if r["name"] in args.groups]
    runs_root = proj(args.runs)
    if args.limit and os.path.abspath(runs_root) == os.path.abspath(proj("runs")):
        # a --limit result in the campaign tree would later be mistaken for a full result and skipped
        raise SystemExit("--limit results must not land in runs/; pass another folder, e.g. --runs runs/smoke")
    for key in args.models:
        if key not in MODELS:
            raise SystemExit(f"model {key} is not in the registry ({', '.join(MODELS)})")
        if not args.list and not os.path.isdir(proj(MODELS[key]["path"])):  # fail fast if a model is not on local disk
            raise SystemExit(f"model {key} is not a local directory: {proj(MODELS[key]['path'])}")

    for m in args.models:
        for r in plan:
            out_dir = os.path.join(runs_root, r["root"], m, r["name"])
            no_think = r["think"] == "on" and not MODELS[m].get("think")
            if args.list:
                state = "skip" if no_think else ("DONE" if done(out_dir) else "todo")
                print(f"{state}  {m:16s} {r['root']:10s} {r['name']:27s} {r['form']} fs={r['shots']:<2d} {r['template']:4s} "
                      f"think={r['think']:3s} bs={resolve_batch(m, r):6s} cap={str(r['max_gen_toks'] or '-'):5s} {r['tasks']}")
                continue
            if no_think:
                print(f"[skip] {m}/{r['root']}/{r['name']}: the model has no think mode", flush=True)
                continue
            if done(out_dir):
                print(f"[skip] {m}/{r['root']}/{r['name']} already has results", flush=True)
                continue
            cmd = build_cmd(m, r, out_dir, args.gpu, args.limit, dp=args.dp)
            print(f"\n[run ] {time.strftime('%m-%d %H:%M:%S')} {m}/{r['root']}/{r['name']}\n       {' '.join(cmd)}", flush=True)
            t0 = time.time()
            res = subprocess.run(cmd, cwd=HARNESS)
            status = "ok" if res.returncode == 0 else f"FAILED rc={res.returncode}"
            print(f"[{status}] {m}/{r['root']}/{r['name']} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
