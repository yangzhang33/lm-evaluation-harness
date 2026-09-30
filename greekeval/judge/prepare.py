"""Build the judge prompt sets from the locally cached ILSP datasets.

    python -m greekeval.judge.prepare            # writes data/judge/<set>.jsonl

One record per item, the same shape for every set so the generator and the judge need no per-set code:
    {"id", "set", "category", "turns": [user turn, ...], "reference": [answer or null per turn],
     "anchors": [{"answer", "comment"}, ...], "meta": {...}}

Sets (nothing is fetched; HF_HUB_OFFLINE stays on):
  mtbench_el   ilsp/mt-bench-greek        80 questions x 2 turns, 8 categories; 39 carry a reference answer
  arenahard_el ilsp/m-ArenaHard_greek     500 single-turn prompts (the `test` translation; `train` is a second,
                                          informal-register translation of the same 500 and is kept in meta)
  civics_el    ilsp/greek_civics_qa       407 native questions with the textbook answer as reference and two
                                          ChatGPT answers + human comments as scoring anchors
`ilsp/vibeeval_greek` is deliberately absent: every item needs an image.
"""
from __future__ import annotations

import ast
import glob
import json
import os

import pyarrow.parquet as pq

from ..paths import proj

HUB = os.path.expanduser("~/.cache/huggingface/hub")
OUT = "data/judge"


def _parquet(name: str, split: str) -> list[dict]:
    fs = glob.glob(f"{HUB}/datasets--ilsp--{name}/snapshots/*/data/{split}-*.parquet")
    assert fs, f"{name}/{split} is not in the local cache"
    return pq.read_table(sorted(fs)[-1]).to_pylist()


def _aslist(x):
    """ILSP stores list columns as their Python repr in some files."""
    if isinstance(x, list):
        return x
    if not x:
        return []
    return ast.literal_eval(x)


def mtbench() -> list[dict]:
    out = []
    for r in _parquet("mt-bench-greek", "train"):
        turns, ref = _aslist(r["turns"]), _aslist(r["reference"])
        out.append({"id": f"mtb-{r['question_id']}", "set": "mtbench_el", "category": r["category"],
                    "turns": turns, "reference": [ref[i] if i < len(ref) else None for i in range(len(turns))],
                    "anchors": [], "meta": {"turns_en": _aslist(r["turns_en"])}})
    return out


def arenahard() -> list[dict]:
    informal = {r["question_id"]: r["prompt"] for r in _parquet("m-ArenaHard_greek", "train")}
    return [{"id": f"ah-{r['question_id']}", "set": "arenahard_el", "category": r["cluster"],
             "turns": [r["prompt"]], "reference": [None], "anchors": [],
             "meta": {"prompt_en": r["prompt_en"], "prompt_informal": informal.get(r["question_id"])}}
            for r in _parquet("m-ArenaHard_greek", "test")]


def civics() -> list[dict]:
    out = []
    for r in _parquet("greek_civics_qa", "default"):
        anchors = [{"answer": r[f"chatGPT_answer_{i}"], "comment": r[f"chatGPT_comment_{i}"]}
                   for i in (1, 2) if r.get(f"chatGPT_answer_{i}")]
        out.append({"id": f"civ-{r['id']}", "set": "civics_el", "category": r["answer_source"],
                    "turns": [r["question"]], "reference": [r["answer"]], "anchors": anchors,
                    "meta": {"question_source": r["question_source"], "answer_comments": r["answer_comments"]}})
    return out


def main() -> None:
    os.makedirs(proj(OUT), exist_ok=True)
    for name, fn in [("mtbench_el", mtbench), ("arenahard_el", arenahard), ("civics_el", civics)]:
        rows = fn()
        with open(proj(f"{OUT}/{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_turns = sum(len(r["turns"]) for r in rows)
        n_ref = sum(1 for r in rows for x in r["reference"] if x)
        print(f"{name}: {len(rows)} items, {n_turns} turns, {n_ref} with reference -> {OUT}/{name}.jsonl")


if __name__ == "__main__":
    main()
