"""Score the collected answers with an LLM judge. Provider-agnostic; `--provider dry` renders every judge prompt to disk
without calling anything, so the whole pipeline can be checked before an API key exists.

    python -m greekeval.judge.judge --provider dry --models <key> [<key> ...]
    python -m greekeval.judge.judge --provider anthropic --judge claude-opus-5 --models ... [--baseline <key>]
    python -m greekeval.judge.judge --summarize --judge claude-opus-5

Four rubrics, kept in separate calls so the dimensions do not contaminate each other:
  answer   ILSP's MT-Bench FlowJudge rubric, verbatim (evaluation/ilsp-assets): 1-5 for "how well the response
           answers", with the reference answer in the criteria when the item has one. MT-Bench turn 2 is judged with
           the full conversation in <inputs>. Civics gets the textbook answer as reference plus the two ChatGPT
           answers with ILSP's human comments as calibration anchors. Sets: mtbench_el (both turns), civics_el.
  greek    Language quality of the Greek itself: fluency/idiomaticity 1-5 and three flags (translationese,
           grammar errors, language switching). This is the dimension no benchmark measures. Sets: mtbench_el turn
           1, arenahard_el, civics_el.
  pairwise Arena-Hard style: the model's answer against a baseline model's answer to the same prompt, judged twice
           with positions swapped; a win needs both orders to agree, a split is a tie. Sets: arenahard_el.
Output: runs/judge/verdicts/<judge>/<set>/<rubric>/<model>.jsonl, one record per (id, turn[, position]); resumable.
Everything the judge reads is untrusted model output; a reply that is not a verdict is stored raw and scored None.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import statistics
import time
from collections import defaultdict

from ..log import get_logger
from ..models import JUDGE_BASELINE
from ..paths import proj

log = get_logger("greekeval.judge.judge")
GEN_ROOT = "runs/judge/generations"
OUT_ROOT = "runs/judge/verdicts"

# ----------------------------------------------------------------------------------------------- rubric texts
_FLOW_INSTRUCTIONS = """# INSTRUCTIONS FOR THE EVALUATION
1. Understand the task and criteria: Familiarize yourself with the task to be evaluated. \
Review the evaluation criteria and scoring rubric to understand the different levels of \
performance and the descriptions for each score.
2. Review the inputs and output: Look at the inputs provided for the task. Examine the output \
generated from completing the task.
3. Compare output to score descriptions: Compare the output against the criteria and score \
descriptions in the scoring rubric. For each criterion,decide which description best matches the \
output.
4. After comparing the output to the score descriptions, pay attention to the small details that \
might impact the final score that you assign. Sometimes a small difference can dictate the final \
score.
5. Write verbal feedback justifying your evaluation that includes a detailed rationale, referring \
to specific aspects of the output and comparing them to the rubric.
6. Assign a final score based on the scoring rubric.

## FORMAT FOR THE EVALUATION
- Write the verbal feedback inside <feedback> tags without any additional surrounding text.
- Write the numeric score inside <score> tags, without any additional surrounding text and always \
after the feedback.

Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric."""

_FLOW_RUBRIC = """<scoring_rubric>
- Score 1: The response completely fails to answer the question.
- Score 2: The response barely answers the question.
- Score 3: The response partially answers the question.
- Score 4: The response mostly answers the question.
- Score 5: The response completely answers the question.
</scoring_rubric>"""


def prompt_answer(inputs: str, answer: str, reference: str | None, anchors: list[dict] | None = None) -> str:
    """ILSP's flow_judge_prompt_mt_bench_el_{with,without}_ref, verbatim apart from the optional anchors block."""
    if reference:
        lang = "the input, response answer and reference answer are all in Greek"
        crit = f"How well the response answers the question, the reference answer is:\n{reference}"
    else:
        lang = "both the input and the response answer are in Greek"
        crit = "How well the response answers the question?"
    if anchors:
        crit += "\n\nFor calibration, here are other answers to the same question with a human expert's comment on each:"
        for i, a in enumerate(anchors, 1):
            crit += f"\n<anchor_{i}>\n{a['answer']}\n</anchor_{i}>\n<anchor_{i}_comment>{a['comment']}</anchor_{i}_comment>"
    return f"""# GOAL
Your job is to evaluate a task carried out by an AI system powered by a large \
language model.

You will be provided with the inputs and output of the task in the Greek language, \
as well as the evaluation criteria and scoring rubric. \
Your task is to evaluate the output of the AI system based on the evaluation \
criteria and scoring rubric provided.

# INPUT
Below are the inputs required for performing the task:
<inputs>
{inputs}
</inputs>

# OUTPUT
Below is the output of the task:
<output>
{answer}
</output>

# EVALUATION CRITERIA AND SCORING RUBRIC
Here are the evaluation criteria and the rubric that you need to use for evaluating the task. \
All evaluation criteria should consider the fact that {lang} and should be evaluated based on their score in Greek:
<evaluation_criteria>
{crit}
</evaluation_criteria>

{_FLOW_RUBRIC}

{_FLOW_INSTRUCTIONS}"""


def prompt_greek(inputs: str, answer: str) -> str:
    return f"""# GOAL
You are a native-level expert in Modern Greek. Judge ONLY the quality of the Greek in the output below — not whether
it answers the question well, not its factual accuracy, not its length. A wrong but beautifully written answer must
score high here; a correct answer in stilted, translated-sounding Greek must score low.

# INPUT (for context only)
<inputs>
{inputs}
</inputs>

# OUTPUT (what you judge)
<output>
{answer}
</output>

# SCORING
Fluency and idiomaticity of the Greek:
- Score 1: Not usable Greek: broken grammar throughout, wrong script, or largely another language.
- Score 2: Understandable but clearly machine-like: word-for-word translation of English structures, wrong
  collocations, unnatural word order in most sentences.
- Score 3: Mostly correct Greek with noticeable translationese: some calques, some unidiomatic phrasing, occasional
  gender/case/agreement slips.
- Score 4: Natural Greek a native reader would accept, with at most a couple of minor awkward spots.
- Score 5: Fully idiomatic, native-quality Greek; nothing marks it as produced by translation or by a model.

Also answer three yes/no flags:
- translationese: does the text read like a translation from English (calques, English syntax, literal idioms)?
- grammar_errors: are there any errors of gender, case, number, agreement, verb form, spelling or accent marks?
- language_switch: does the text switch into another language or script anywhere (excluding names, code, or
  terms the prompt itself asked for in another language)?

# FORMAT
Write brief feedback inside <feedback> tags, quoting one or two concrete phrases as evidence.
Then write exactly these four lines, nothing else after them:
<score>N</score>
<translationese>yes|no</translationese>
<grammar_errors>yes|no</grammar_errors>
<language_switch>yes|no</language_switch>"""


def prompt_pairwise(question: str, answer_a: str, answer_b: str) -> str:
    return f"""Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to
the user prompt displayed below. Both the prompt and the responses are in Greek. You will be given assistant A's
answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging
any answers. When evaluating the assistants' answers, compare both assistants' answers with your answer. You must
identify and correct any mistakes or inaccurate information. Then consider if the assistant's answers are helpful,
relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions.
Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means
the response is clear and not verbose or excessive. Consider the quality of the Greek as part of helpfulness: an
answer in unnatural or partly non-Greek text is less helpful to a Greek user. Do not favor the longer answer for
its length alone. Avoid any position bias and ensure that the order in which the responses were presented does not
influence your decision.

After providing your explanation, output your final verdict by strictly following this format:
"[[A>>B]]" (A significantly better), "[[A>B]]" (A slightly better), "[[A=B]]" (tie),
"[[B>A]]" (B slightly better), "[[B>>A]]" (B significantly better).

<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>"""


# ----------------------------------------------------------------------------------------------- providers
class Provider:
    def __init__(self, name: str, judge_model: str, out_dir: str):
        self.name, self.model, self.out_dir = name, judge_model, out_dir
        self.client = None
        if name == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif name == "openai":
            import openai
            self.client = openai.OpenAI()
        elif name != "dry":
            raise SystemExit(f"unknown provider {name}")

    def complete(self, prompt: str, tag: str) -> str:
        if self.name == "dry":
            os.makedirs(f"{self.out_dir}/_dry_prompts", exist_ok=True)
            with open(f"{self.out_dir}/_dry_prompts/{tag}.txt", "w", encoding="utf-8") as fh:
                fh.write(prompt)
            return "<feedback>dry run</feedback><score>0</score><translationese>no</translationese>" \
                   "<grammar_errors>no</grammar_errors><language_switch>no</language_switch>[[A=B]]"
        for attempt in range(6):
            try:
                if self.name == "anthropic":
                    r = self.client.messages.create(model=self.model, max_tokens=1500, temperature=0,
                                                    messages=[{"role": "user", "content": prompt}])
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                r = self.client.chat.completions.create(model=self.model, temperature=0, max_tokens=1500,
                                                        messages=[{"role": "user", "content": prompt}])
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limits, transient network: back off and retry
                wait = 2 ** attempt
                log.warning("%s: %s; retrying in %ds", tag, str(e)[:120], wait)
                time.sleep(wait)
        raise RuntimeError(f"judge call failed six times: {tag}")


# ----------------------------------------------------------------------------------------------- parsing
def parse_score(text: str) -> int | None:
    m = re.findall(r"<score>\s*(\d+)\s*</score>", text)
    return int(m[-1]) if m else None


def parse_flags(text: str) -> dict:
    out = {}
    for k in ("translationese", "grammar_errors", "language_switch"):
        m = re.findall(rf"<{k}>\s*(yes|no)\s*</{k}>", text, flags=re.I)
        out[k] = (m[-1].lower() == "yes") if m else None
    return out


def parse_pairwise(text: str) -> str | None:
    m = re.findall(r"\[\[(A>>B|A>B|A=B|B>A|B>>A)\]\]", text)
    return m[-1] if m else None


# ----------------------------------------------------------------------------------------------- data access
def load_items(set_name: str) -> dict[str, dict]:
    return {r["id"]: r for r in (json.loads(l) for l in open(proj(f"data/judge/{set_name}.jsonl"), encoding="utf-8"))}


def load_generations(model: str, set_name: str) -> dict[str, dict]:
    p = proj(f"{GEN_ROOT}/{model}/{set_name}.jsonl")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in open(p, encoding="utf-8"))}


def conversation(item: dict, gen: dict, upto_turn: int) -> str:
    """The exchange as the judge sees it: every user turn up to `upto_turn`, with the model's earlier answers."""
    parts = []
    for t in range(upto_turn + 1):
        parts.append(f"[Χρήστης]\n{item['turns'][t]}")
        if t < upto_turn:
            parts.append(f"[Βοηθός]\n{gen['responses'][t]}")
    return "\n\n".join(parts)


class Store:
    """Append-only jsonl per (judge, set, rubric, model); keyed on (id, turn, position) for resumption."""
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.done = {}
        if os.path.exists(path):
            for l in open(path, encoding="utf-8"):
                r = json.loads(l); self.done[(r["id"], r.get("turn", 0), r.get("position", ""))] = r

    def has(self, id_, turn=0, position=""):
        return (id_, turn, position) in self.done

    def add(self, rec: dict):
        self.done[(rec["id"], rec.get("turn", 0), rec.get("position", ""))] = rec
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------------------------- rubric runners
def run_answer(prov: Provider, model: str, set_name: str, limit: int | None) -> None:
    items, gens = load_items(set_name), load_generations(model, set_name)
    store = Store(f"{prov.out_dir}/{set_name}/answer/{model}.jsonl")
    ids = [i for i in items if i in gens][:limit]
    n = 0
    for id_ in ids:
        it, g = items[id_], gens[id_]
        for t, resp in enumerate(g["responses"]):
            if store.has(id_, t):
                continue
            anchors = it["anchors"] if set_name == "civics_el" else None
            p = prompt_answer(conversation(it, g, t), resp, it["reference"][t], anchors)
            raw = prov.complete(p, f"answer-{set_name}-{model}-{id_}-t{t}")
            store.add({"id": id_, "turn": t, "score": parse_score(raw), "raw": raw, "category": it["category"]})
            n += 1
    log.info("answer/%s/%s: %d new verdicts (%d items with generations)", set_name, model, n, len(ids))


def run_greek(prov: Provider, model: str, set_name: str, limit: int | None) -> None:
    items, gens = load_items(set_name), load_generations(model, set_name)
    store = Store(f"{prov.out_dir}/{set_name}/greek/{model}.jsonl")
    ids = [i for i in items if i in gens][:limit]
    n = 0
    for id_ in ids:
        it, g = items[id_], gens[id_]
        if store.has(id_, 0) or not g["responses"][0].strip():
            continue
        raw = prov.complete(prompt_greek(it["turns"][0], g["responses"][0]), f"greek-{set_name}-{model}-{id_}")
        store.add({"id": id_, "turn": 0, "score": parse_score(raw), **parse_flags(raw), "raw": raw,
                   "category": it["category"]})
        n += 1
    log.info("greek/%s/%s: %d new verdicts", set_name, model, n)


def run_pairwise(prov: Provider, model: str, baseline: str, set_name: str, limit: int | None) -> None:
    if model == baseline:
        return
    items, gens, base = load_items(set_name), load_generations(model, set_name), load_generations(baseline, set_name)
    store = Store(f"{prov.out_dir}/{set_name}/pairwise_vs_{baseline}/{model}.jsonl")
    ids = [i for i in items if i in gens and i in base][:limit]
    n = 0
    for id_ in ids:
        q = items[id_]["turns"][0]; mine, theirs = gens[id_]["responses"][0], base[id_]["responses"][0]
        for pos, (a, b) in (("model_A", (mine, theirs)), ("model_B", (theirs, mine))):
            if store.has(id_, 0, pos):
                continue
            raw = prov.complete(prompt_pairwise(q, a, b), f"pair-{set_name}-{model}-vs-{baseline}-{id_}-{pos}")
            store.add({"id": id_, "turn": 0, "position": pos, "verdict": parse_pairwise(raw), "raw": raw,
                       "category": items[id_]["category"]})
            n += 1
    log.info("pairwise/%s/%s vs %s: %d new verdicts", set_name, model, baseline, n)


# ----------------------------------------------------------------------------------------------- summary
def _win_value(verdict: str | None, model_is_a: bool) -> float | None:
    if verdict is None:
        return None
    if verdict == "A=B":
        return 0.5
    a_wins = verdict.startswith("A")
    return 1.0 if a_wins == model_is_a else 0.0


def summarize(out_dir: str) -> str:
    rows = []
    for f in sorted(glob.glob(f"{out_dir}/*/*/*.jsonl")):
        set_name, rubric, model = f[len(out_dir) + 1:-len(".jsonl")].split("/")
        recs = [json.loads(l) for l in open(f, encoding="utf-8")]
        if rubric == "answer":
            by_turn = defaultdict(list)
            for r in recs:
                if r["score"] is not None:
                    by_turn[r["turn"]].append(r["score"])
            for t, sc in sorted(by_turn.items()):
                rows.append((set_name, f"answer score 1-5, turn {t + 1}", model, f"{statistics.mean(sc):.2f}", len(sc)))
        elif rubric == "greek":
            sc = [r["score"] for r in recs if r["score"] is not None]
            if sc:
                rows.append((set_name, "greek fluency 1-5", model, f"{statistics.mean(sc):.2f}", len(sc)))
            for k in ("translationese", "grammar_errors", "language_switch"):
                v = [r[k] for r in recs if r.get(k) is not None]
                if v:
                    rows.append((set_name, f"{k} rate", model, f"{sum(v) / len(v):.1%}", len(v)))
        elif rubric.startswith("pairwise_vs_"):
            per_id = defaultdict(dict)
            for r in recs:
                per_id[r["id"]][r["position"]] = _win_value(r["verdict"], r["position"] == "model_A")
            vals = [(d.get("model_A"), d.get("model_B")) for d in per_id.values()]
            vals = [(a, b) for a, b in vals if a is not None and b is not None]
            if vals:
                wr = statistics.mean((a + b) / 2 for a, b in vals)
                strict = statistics.mean(1.0 if a == b == 1.0 else 0.0 for a, b in vals)
                rows.append((set_name, f"win rate vs {rubric[len('pairwise_vs_'):]} (ties=0.5)", model, f"{wr:.1%}", len(vals)))
                rows.append((set_name, f"  · wins in both orders", model, f"{strict:.1%}", len(vals)))
    lines = ["| set | metric | model | value | n |", "|---|---|---|---|---|"]
    lines += [f"| {s} | {m} | {mo} | {v} | {n} |" for s, m, mo, v, n in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai", "dry"], default="dry")
    ap.add_argument("--judge", default="dry", help="judge model name; also names the output directory")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--sets", nargs="*", default=["mtbench_el", "arenahard_el", "civics_el"])
    ap.add_argument("--rubrics", nargs="*", default=["answer", "greek", "pairwise"])
    ap.add_argument("--baseline", default=JUDGE_BASELINE,
                    help="opponent for the pairwise rubric (default: `judge_baseline` in the registry)")
    ap.add_argument("--limit", type=int, help="items per (model, set); for probes")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    out_dir = proj(f"{OUT_ROOT}/{args.judge}")
    if args.summarize:
        print(summarize(out_dir)); return
    prov = Provider(args.provider, args.judge, out_dir)
    for model in args.models:
        for s in args.sets:
            if "answer" in args.rubrics and s in ("mtbench_el", "civics_el"):
                run_answer(prov, model, s, args.limit)
            if "greek" in args.rubrics:
                run_greek(prov, model, s, args.limit)
            if "pairwise" in args.rubrics and s == "arenahard_el" and args.baseline:
                run_pairwise(prov, model, args.baseline, s, args.limit)


if __name__ == "__main__":
    main()
