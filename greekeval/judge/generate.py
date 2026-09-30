"""Collect model answers for the judge prompt sets, through each model's own chat path.

    python -m greekeval.judge.generate --gpu 0 --models <key> [<key> ...] [--sets mtbench_el ...] [--limit 3]

Reads data/judge/<set>.jsonl (see judge.prepare), writes runs/judge/generations/<model>/<set>.jsonl, one record per
item: {"id","set","model","turns","responses","n_new_tokens","closed","gen"}. Resumable: items already in the output
file are skipped.

Generation path is the campaign's: every model through its own chat template with a closed, empty think block
(`think: enable_thinking` -> `enable_thinking=False`; `think: suffix` -> the closing tag appended and
`reasoning_content=""` on history), stop at the model's turn end, greedy. Multi-turn sets feed the model's own
cleaned turn-1 answer back as history.
The token budget should be character-aligned across tokenizers: MT-Bench's 1024 tokens buy a Greek-efficient
tokenizer about 5,000 characters, and a tokenizer that spends more tokens per Greek character needs a larger budget
to write the same text, or the judge scores the tokenizer. Set it per model in the registry (`judge_max_new`).
A model marked `judge: false` in the registry (it cannot follow its chat template) is refused: its output under a
template is a protocol artifact, not an answer.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..log import get_logger
from ..models import MODELS
from ..paths import proj
from ..templates import TemplateSpec, prepare_messages

log = get_logger("greekeval.judge.generate")
SETS = ["mtbench_el", "arenahard_el", "civics_el"]
DEFAULT_MAX_NEW = 1024      # a model's own budget: `judge_max_new` in the registry, see docstring
K2_CLOSE = "</ifm|think>"   # the closing tag of `think: suffix` templates


def load(model_key: str, gpu: int):
    m = MODELS[model_key]
    trc = bool(m.get("trust_remote_code"))
    tok = AutoTokenizer.from_pretrained(proj(m.get("tokenizer") or m["path"]), trust_remote_code=trc)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(proj(m["path"]), dtype=torch.bfloat16, device_map=f"cuda:{gpu}",
                                                 trust_remote_code=trc).eval()
    spec = TemplateSpec(assistant_start="", turn_end=m["turn_end"], add_reasoning_field=(m.get("think") == "suffix"))
    return tok, model, spec, m


def build_prompt(tok, spec: TemplateSpec, m: dict, messages: list[dict]) -> str:
    kw = {"enable_thinking": False} if m.get("think") == "enable_thinking" else {}
    p = tok.apply_chat_template(prepare_messages(messages, spec), tokenize=False, add_generation_prompt=True, **kw)
    return p + K2_CLOSE if m.get("think") == "suffix" else p


def clean(text: str, spec: TemplateSpec, tok) -> tuple[str, bool]:
    closed = spec.turn_end in text or (tok.eos_token and tok.eos_token in text)
    for stop in (spec.turn_end, tok.eos_token):
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    for marker in (K2_CLOSE, "</think>"):        # a think block the model re-opened despite the prefix
        if marker in text:
            text = text.split(marker, 1)[1]
    return text.strip(), closed


@torch.no_grad()
def generate(tok, model, spec, m, prompts: list[str], max_new: int, bs: int) -> list[tuple[str, int, bool]]:
    end_ids = sorted({i for i in (tok.convert_tokens_to_ids(spec.turn_end), tok.eos_token_id) if i is not None})
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))   # long first: OOM shows up immediately
    out: list = [None] * len(prompts)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             eos_token_id=end_ids, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row, ids in zip(idx, new):
            ids = ids.tolist()
            while ids and ids[-1] == tok.pad_token_id:
                ids.pop()
            text, closed = clean(tok.decode(ids, skip_special_tokens=False), spec, tok)
            out[row] = (text, len(ids), closed)
    return out


def run_set(tok, model, spec, m, model_key: str, set_name: str, out_root: str, limit: int | None, bs: int) -> None:
    items = [json.loads(l) for l in open(proj(f"data/judge/{set_name}.jsonl"), encoding="utf-8")]
    if limit:
        items = items[:limit]
    out_path = proj(f"{out_root}/{model_key}/{set_name}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8")}
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        log.info("%s/%s: all %d items present, skipping", model_key, set_name, len(items)); return
    max_new = m.get("judge_max_new", DEFAULT_MAX_NEW)
    n_turns = max(len(it["turns"]) for it in todo)
    hist = [[] for _ in todo]
    responses = [[] for _ in todo]; ntoks = [[] for _ in todo]; closed = [[] for _ in todo]
    t0 = time.time()
    for t in range(n_turns):
        live = [i for i, it in enumerate(todo) if t < len(it["turns"])]
        for i in live:
            hist[i].append({"role": "user", "content": todo[i]["turns"][t]})
        prompts = [build_prompt(tok, spec, m, hist[i]) for i in live]
        res = generate(tok, model, spec, m, prompts, max_new, bs)
        for i, (text, n, c) in zip(live, res):
            hist[i].append({"role": "assistant", "content": text})
            responses[i].append(text); ntoks[i].append(n); closed[i].append(c)
        log.info("%s/%s turn %d: %d prompts in %.1f min", model_key, set_name, t + 1, len(live), (time.time() - t0) / 60)
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, it in enumerate(todo):
            fh.write(json.dumps({"id": it["id"], "set": set_name, "model": model_key, "turns": it["turns"],
                                 "responses": responses[i], "n_new_tokens": ntoks[i], "closed": closed[i],
                                 "gen": {"max_new_tokens": max_new, "greedy": True}}, ensure_ascii=False) + "\n")
    log.info("%s/%s: wrote %d items (%d already present) in %.1f min -> %s",
             model_key, set_name, len(todo), len(done), (time.time() - t0) / 60, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=SETS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_root", default="runs/judge/generations")
    args = ap.parse_args()
    for key in args.models:
        if MODELS[key].get("judge") is False:
            raise SystemExit(f"{key} is marked `judge: false` in the registry: its template output is an artifact, "
                             "not an answer")
        if not os.path.isdir(proj(MODELS[key]["path"])):
            raise SystemExit(f"{key} is not a local directory: {MODELS[key]['path']}")
    for key in args.models:
        tok, model, spec, m = load(key, args.gpu)
        for s in args.sets:
            run_set(tok, model, spec, m, key, s, args.out_root, args.limit, args.batch_size)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
