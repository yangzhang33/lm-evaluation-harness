"""Drive a Greek eval campaign: every model x every task group, resumable.

    python -m greekeval.run_suite --gpu 0 --models <key> [<key> ...]
    python -m greekeval.run_suite --list            # show the plan and exit

Models come from the project's registry (see models.py). Protocol (see GREEK_EVAL.md next to this package):
  - multiple choice -> raw few-shot likelihood, no chat template, for ALL models. This is what published Greek
    numbers use, so base and fine-tuned stay comparable.
  - generative      -> chat template for ALL models, so the path is identical on both sides. A base model that
    cannot follow its own template runs these groups with --raw_gen instead.
Results land in runs/eval/<model_key>/<group>/ ; a group that already has a results json is skipped.
Relative paths (model paths in the registry, --out_root) are taken from the project root, not from the current
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
from .paths import HARNESS, proj

# (group, tasks, num_fewshot, chat_template, batch_size[, per_task_limit])
# Every group runs its full test set. done() skips a group that already has results, so nothing is re-run unless
# its folder is deleted; a result made with --limit carries its cap in the `limit` column of all_scores.csv.
GROUPS = [
    ("mcq_fs5",    "greekmmlu,belebele_ell_Grek,ilspgreekasep",                 5,  False, "4"),
    ("mcq_fs5_en", "mmlu",                                                      5,  False, "4"),
    ("gen_fs0",    "ilspgreekifeval,ifeval",                                    0,  True,  "8"),
    ("gen_civics", "ilspgreekcivicsqa",                                         0,  True,  "8"),  # keeps its own until/100-token cap
    ("mcq_fs0",    "greekmmlu,ilspgreektruthfulqa_mc1,ilspgreektruthfulqa_mc2", 0,  False, "8"),
    ("mcq_fs15",   "ilspgreekmedicalmcqa",                                      15, False, "2"),
    ("gen_fs8",    "ilspgreekmgsm",                                             8,  True,  "8"),   # superseded, see gen_fs8b
    # gen_fs8 inherits lm-eval's default max_gen_toks=256, which cuts Greek chain-of-thought answers off (Greek
    # costs many tokenizers 2-4x more tokens than English), so its score measures the cap. gen_fs8b is the same task
    # with a cap the answer cannot hit; it replaces gen_fs8. Kept both so old runs stay reproducible.
    ("gen_fs8b",   "ilspgreekmgsm",                                             8,  True,  "8"),
    ("mcq_fs5b",   "ilspgreekmmlu",                                             5,  False, "4"),
    ("mcq_fs10",   "ilspgreekhellaswag",                                        10, False, "4"),
    ("mcq_fs25",   "ilspgreekarc_easy,ilspgreekarc_challenge",                  25, False, "2"),
    ("mcq_fs5w",   "ilspgreekwinogrande",                                       5,  False, "8"),  # filtered pairs, short
    ("gen_trans",  "ilspgreekflores_en_el,ilspgreekflores_el_en,ilspgreektruthfulqa_gen", 0, True, "8"),
    # ilspgreekmmlupro is deliberately excluded from the default plan: 12,032 generative items at up to 2048 new
    # tokens each would dominate the whole campaign, and it is machine-translated. Run it explicitly when wanted:
    #   python -m greekeval.run_suite --groups gen_mmlupro
    ("gen_mmlupro", "ilspgreekmmlupro",                                         0,  True,  "8"),
    # Generative re-scoring of the four letter-scored MCQ tasks: the model writes the option letter instead of the
    # harness ranking " Α"/" Β"/" Γ"/" Δ". Raw prompt like the MCQ groups. Not in the default plan; compare each with
    # its log-likelihood group at the same shot count: gen_mmlu_fs5<->mcq_fs5, gen_mmlu_fs0<->mcq_fs0 (boxed
    # instruction, see the task README), gen_ilsp_fs5<->mcq_fs5b, gen_bele_fs5<->mcq_fs5, gen_mmlu_en_fs5<->mcq_fs5_en.
    ("gen_mmlu_fs5",    "greekmmlu_gen",                                        5,  False, "8"),
    ("gen_mmlu_fs0",    "greekmmlu_gen_boxed",                                  0,  False, "8"),
    ("gen_ilsp_fs5",    "ilspgreekmmlu_gen",                                    5,  False, "8"),
    ("gen_bele_fs5",    "belebele_ell_Grek_gen",                                5,  False, "8"),
    ("gen_mmlu_en_fs5", "mmlu_generative",                                      5,  False, "8"),
]
# the letter-writing re-scoring groups (raw prompt, model writes the option letter); `--chat_mcq` reruns them with the
# chat template into `<group>_chat/`, the mirror image of `--raw_gen`'s `<group>_raw/`
LETTER_GROUPS = {"gen_mmlu_fs5", "gen_mmlu_fs0", "gen_ilsp_fs5", "gen_bele_fs5", "gen_mmlu_en_fs5"}
NON_DEFAULT_GROUPS = {"gen_mmlupro", "gen_mmlu_fs5", "gen_mmlu_fs0", "gen_ilsp_fs5", "gen_bele_fs5", "gen_mmlu_en_fs5",
                      "gen_fs8"}   # gen_fs8: superseded by gen_fs8b (256-token cap artifact), kept for reproducibility
DEFAULT_GROUPS = [g[0] for g in GROUPS if g[0] not in NON_DEFAULT_GROUPS]
# Civics QA (ILSP lighteval) caps generation at 100 tokens and stops at the first newline. Under a chat template the
# newline stop is a raw-completion artifact: a model that opens its answer with "\n" after the think block scores
# empty on every row. We keep the 100-token cap (the substantive constraint) and stop at the turn end instead.
EXTRA_GEN_KWARGS = {"gen_civics": {"max_gen_toks": 100}, "gen_fs8b": {"max_gen_toks": 2048}}
# Caps that apply only with the chat template. A chat-tuned model explains before it writes \boxed{}, so the task's
# 64 tokens cut it off before the box; a model that answers directly stops after a few tokens, so the larger cap
# costs it nothing.
CHAT_GEN_KWARGS = {"gen_mmlu_fs0": {"max_gen_toks": 1024}}
OUT_ROOT = proj("runs/eval")


def done(model: str, group: str) -> bool:
    d = os.path.join(OUT_ROOT, model, group)
    return any(f.startswith("results_") for _, _, fs in os.walk(d) for f in fs) if os.path.isdir(d) else False


def resolve_batch(model: str, group: str, bs: str) -> str:
    """The fixed sizes above fit a 4B model on a 48GB card; bigger models OOM at them. Two policies, set per model
    in the registry (`mcq_batch`):

    `auto` hands sizing to lm-eval (`auto:4`, re-detected four times per run). It works when context lengths are
    uniform and is NOT safe when they are not: it sizes the batch on the contexts it sees first and then OOMs on the
    long tail of few-shot prompts. `half` just halves the group size. Batch size does not change a likelihood score,
    only the time and memory to compute it."""
    policy = MODELS[model].get("mcq_batch")
    # the letter-writing generative groups carry the same 5-shot MMLU-style contexts as the mcq groups, so the same
    # per-model policy applies to them
    if not policy or not (group.startswith("mcq") or group in LETTER_GROUPS):
        return bs
    return "auto:4" if policy == "auto" else str(max(1, int(bs) // 2))


def build_cmd(model: str, group: str, tasks: str, fs: int, chat: bool, bs: str, gpu: int, limit: int | None,
              out_group: str | None = None, think: bool = False, dp: int = 1) -> list[str]:
    m = MODELS[model]
    out_group = out_group or group
    bs = resolve_batch(model, group, bs)
    margs = {"pretrained": os.path.abspath(proj(m["path"])), "dtype": "bfloat16"}
    if m.get("revision"):
        margs["revision"] = m["revision"]
    if m.get("trust_remote_code"):
        margs["trust_remote_code"] = True
    if m.get("tokenizer"):
        margs["tokenizer"] = os.path.abspath(proj(m["tokenizer"]))
    if chat:
        # The generation prompt carries an empty, CLOSED think block (the prefix a non-thinking fine-tune is trained
        # with). Two template styles, named by the registry's `think` field:
        #   enable_thinking (Qwen): enable_thinking=False makes the template emit "<think>\n\n</think>\n\n" after the
        #                           assistant header.
        #   suffix (K2):            the template has no switch and always opens "<ifm|think>\n", so the closing tag is
        #                           appended verbatim.
        # Without this a reasoning model thinks at length and the generative metrics score the reasoning, not the
        # answer. A model without a think block only needs its own turn-end stop.
        # --think reverses this: the block is left OPEN so the model reasons first. It is a diagnostic, not the
        # campaign protocol; results must go to a separate --out_root, never next to the campaign.
        if m.get("think") == "enable_thinking":
            margs["enable_thinking"] = bool(think)
            if think:
                margs["think_end_token"] = "</think>"          # harness strips the block before scoring
        elif m.get("think") == "suffix":
            if think:
                margs["think_end_token"] = "</ifm|think>"
            else:
                margs["chat_template_suffix"] = "</ifm|think>"
    # --dp N: one lm-eval run data-parallel over GPUs 0..N-1 (accelerate splits the docs across ranks and rank 0 writes
    # a single results file and the gathered samples), so the output folder is the same as a one-GPU run
    launcher = [sys.executable, "-m", "lm_eval"] if dp == 1 else \
        [os.path.join(os.path.dirname(sys.executable), "accelerate"), "launch", "--multi_gpu", "--num_processes", str(dp),
         "--num_machines", "1", "--mixed_precision", "no", "--dynamo_backend", "no", "--main_process_port", "29533",
         "-m", "lm_eval"]
    cmd = launcher + ["--model", "hf", "--model_args", json.dumps(margs), "--tasks", tasks,
           "--num_fewshot", str(fs), "--batch_size", bs] + (["--device", f"cuda:{gpu}"] if dp == 1 else []) + [
           "--output_path", os.path.abspath(os.path.join(OUT_ROOT, model, out_group)), "--log_samples",
           "--seed", "1234"]
    if chat:
        cmd.append("--apply_chat_template")
        # stop at the chat turn end. A base snapshot may ship no generation_config and a tokenizer eos that is not
        # the turn end, so without this it would run to max_gen_toks; it also replaces task-level `until` lists such
        # as MGSM's ["\n\n", "\n"], which cut a multi-line chat answer after its first line.
        gk = {"until": [m["turn_end"]]} | EXTRA_GEN_KWARGS.get(group, {}) | CHAT_GEN_KWARGS.get(group, {})
        if think:
            gk["max_gen_toks"] = 3072          # room for the reasoning block plus the answer
        cmd += ["--gen_kwargs", json.dumps(gk)]
        # A `suffix`-style template rejects assistant messages without a `reasoning_content` field, which is exactly
        # what --fewshot_as_multiturn builds, so such a model gets concatenated few-shot inside one user turn instead.
        if fs > 0:
            # this fork auto-enables multi-turn few-shot with a chat template; the opt-out must be explicit
            cmd += ["--fewshot_as_multiturn", "false" if m.get("think") == "suffix" else "true"]
    elif group in EXTRA_GEN_KWARGS:
        # raw completion keeps the task's own `until`; lm-eval merges --gen_kwargs into it. The cap must follow the
        # group, not the protocol: without this branch a raw run would silently fall back to the 256-token default.
        cmd += ["--gen_kwargs", json.dumps(EXTRA_GEN_KWARGS[group])]
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd


def main() -> None:
    global OUT_ROOT
    # less fragmentation near the memory limit; PyTorch's own OOM message recommends it
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dp", type=int, default=1, help="data-parallel over GPUs 0..N-1 in one run (ignores --gpu)")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    ap.add_argument("--limit", type=int, help="cap items per task (for timing probes)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out_root", default="runs/eval", help="write results here (use a scratch dir for --limit probes)")
    ap.add_argument("--think", action="store_true",
                    help="generative groups with the think block OPEN (Qwen enable_thinking=True, K2 without the closing"
                         " suffix) and max_gen_toks=3072. Diagnostic only: use with --out_root runs/eval_think.")
    ap.add_argument("--suffix", default="",
                    help="appended to every output group folder after the other renames, e.g. _closed; lets two"
                         " conditions of the same group live side by side under one --out_root")
    ap.add_argument("--chat_mcq", action="store_true",
                    help="run the letter-writing MCQ groups WITH each model's chat template (closed think block, own turn"
                         " end, few-shot as multi-turn where the template allows it). Results land in <group>_chat/;"
                         " never put them in the same column as the raw-prompt numbers.")
    ap.add_argument("--raw_gen", action="store_true",
                    help="evaluate the generative groups as raw few-shot completion: no chat template, and each task's"
                         " own stop rules instead of ours. This is the native ILSP/lighteval configuration and the"
                         " standard way to score a pure base model, but it is a DIFFERENT protocol from the rest of"
                         " the campaign -- results land in <group>_raw/ and must not be put in the same column as"
                         " chat-template numbers.")
    args = ap.parse_args()
    OUT_ROOT = proj(args.out_root)
    if (args.think or args.limit) and os.path.abspath(OUT_ROOT) == os.path.abspath(proj("runs/eval")) and not args.list:
        # a --limit run in the campaign tree would later be mistaken for a full result by done()
        if args.limit and not args.think:
            raise SystemExit("--limit results must not land in runs/eval; pass another --out_root")
    if args.think and os.path.abspath(OUT_ROOT) == os.path.abspath(proj("runs/eval")):
        raise SystemExit("--think results must not land in runs/eval; pass --out_root runs/eval_think")

    plan = [(m, tuple(g) + ((None,) if len(g) == 5 else ())) for m in args.models for g in GROUPS if g[0] in args.groups]
    if args.list:
        for m, (g, t, fs, chat, bs, lim) in plan:
            print(f"{'DONE' if done(m, g) else 'todo'}  {m:10s} {g:12s} fs={fs:<3d} chat={int(chat)} bs={bs:6s} "
                  f"limit={lim or '-':6} {t}")
        return

    for key in args.models:                       # fail fast if any model is not on local disk
        pth = proj(MODELS[key]["path"])
        if not os.path.isdir(pth):
            raise SystemExit(f"model {key} is not a local directory: {pth}")
    for m, (g, t, fs, chat, bs, lim) in plan:
        out_g = g
        if args.raw_gen:
            if not chat:               # multiple choice is already template-free: it would be the same run twice
                print(f"[skip] {m}/{g} is not a generative group", flush=True)
                continue
            chat, out_g = False, g + "_raw"
        # every variant flag must rename out_g BEFORE the done() check below; on 2026-09-25 the --chat_mcq rename sat
        # after it, done() looked at the raw-prompt folder, found results, and the whole chat step silently skipped
        # (--limit probes bypass done() and did not catch it)
        if args.chat_mcq:
            if g not in LETTER_GROUPS:
                print(f"[skip] {m}/{g} is not a letter-writing MCQ group", flush=True); continue
            chat, out_g = True, g + "_chat"
        out_g += args.suffix
        if done(m, out_g) and not args.limit:
            print(f"[skip] {m}/{out_g} already has results", flush=True)
            continue
        if args.think and not chat:
            print(f"[skip] {m}/{g} is not a generative group", flush=True); continue
        cmd = build_cmd(m, g, t, fs, chat, bs, args.gpu, args.limit or lim, out_group=out_g, think=args.think, dp=args.dp)
        print(f"\n[run ] {time.strftime('%m-%d %H:%M:%S')} {m}/{out_g}\n       {' '.join(cmd)}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=HARNESS)
        status = "ok" if r.returncode == 0 else f"FAILED rc={r.returncode}"
        print(f"[{status}] {m}/{out_g} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
