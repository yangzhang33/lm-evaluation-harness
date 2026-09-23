# Greek evaluation suite

> 中文版：[`GREEK_EVAL.zh.md`](GREEK_EVAL.zh.md)

A complete, runnable Greek LLM evaluation stack built on lm-evaluation-harness: 13 Greek benchmarks across 14 task
directories (206 leaf tasks), plus a rule-based diagnostic for generation quality. Everything runs offline once the
datasets are cached, and every task records how it differs from its reference implementation.

**Status: built and verified.** Every task below was executed end to end against a real model
(`ilsp/Meltemi-7B-Instruct-v1.5`, and `models/e1_qwen3.5-4b-base_*_merged` for the first batch) — not just loaded.
Running the actual evaluation campaign is a separate job; the training repo's `evaluation/TASKS_TODO.md` records what
is deliberately left out and why.

## Layout

```
lm_eval/tasks/
├── greekmmlu/          GreekMMLU — natively sourced, upstream since PR #3581
└── ilspgreek*/         the eleven task families added here, one directory each with its own README
```

Each task directory's README records how that task differs from its reference implementation, including the bugs
found in the reference.

Two pieces of this suite live outside the harness, in the training repo that wraps it:

- `src/greekllm/eval/failure_profile.py` — the generation diagnostic described below.
- `evaluation/ilsp-assets/` — ILSP's Greek MT-Bench judge prompts, kept for a judge script that does not fit lm-eval.
- `evaluation/TASKS_TODO.md` — what is left to build, what is blocked, and what was evaluated and rejected.

## Install

```bash
pip install -e .            # from this directory                    # adds ~28 packages, touches no torch/transformers pin
pip install langdetect immutabledict # ilspgreekifeval
python -c "import nltk; nltk.download('punkt_tab')"  # Greek sentence splitting, ~11 MB
```

`ilspgreektruthfulqa_gen` additionally needs `sacrebleu` and `rouge_score` (both pulled in by the install above).

## Tasks built here

Native = written in Greek by Greek speakers. MT = machine-translated from English, so both translation artifacts and
contamination risk apply.

| task | dataset | type | eval set | source | ILSP few-shot |
|---|---|---|---|---|---|
| `greekmmlu` | `dascim/GreekMMLU` | MCQ | 16,857 public, 45 subjects | **native** | — |
| `greekmmlu_gen` | `dascim/GreekMMLU` | generative, writes the letter; prompt identical to `greekmmlu` | 16,632, 45 subjects | **native** | 5 |
| `greekmmlu_gen_boxed` | `dascim/GreekMMLU` | generative, `\boxed{}` letter with a format instruction | 16,632, 45 subjects | **native** | 0 |
| `ilspgreekmmlu` | `ilsp/mmlu_greek` | MCQ | 14,042, 57 subjects | MT | 5 |
| `ilspgreekmmlupro` | `ilsp/MMLU-Pro_greek` | generative, 10-way | 12,032 | MT | 0 |
| `ilspgreekarc_easy` / `_challenge` | `ilsp/arc_greek` | MCQ | 2,376 / 1,168 | MT | 25 |
| `ilspgreekhellaswag` | `ilsp/hellaswag_greek` | MCQ | 10,024 | MT | 10 |
| `ilspgreektruthfulqa_mc1` / `_mc2` | `ilsp/truthful_qa_greek` | MCQ | 817 | MT, human-checked | 0 |
| `ilspgreektruthfulqa_gen` | `ilsp/truthful_qa_greek` | generative | 817 | MT, human-checked | 0 |
| `ilspgreekmedicalmcqa` | `ilsp/medical_mcqa_greek` | MCQ, 5-way | 1,602 | **native** (DOATAP exams) | 15 |
| `ilspgreekasep` | `ilsp/mcqa_greek_asep` | MCQ | 2,346 | **native** (ASEP exams) | — |
| `ilspgreekifeval` | `ilsp/ifeval_greek` | generative, rule-scored | 541 | translated + localised | 0 |
| `ilspgreekmgsm` | `ilsp/mgsm_greek` | generative, CoT | 250 | MT | 8 |
| `ilspgreekcivicsqa` | `ilsp/greek_civics_qa` | generative, BLEU | 407 | **native** (textbooks) | 0 |
| `ilspgreekflores_en_el` / `_el_en` | `ilsp/flores200_en-el` | translation | 1,012 each | professional translation | — |

Groups and tags: `ilspgreekmmlu` aggregates its 57 subjects; `greekmmlu` aggregates 45 subjects and also has
`greekmmlu_stem` / `_humanities` / `_social_sciences` / `_other`. Tags `ilspgreekarc`, `ilspgreektruthfulqa` and
`ilspgreekflores` run their members together.

Counted as benchmarks that is **7 multiple-choice and 6 generative** — a far more even split than the task count
suggests, since the two MMLUs alone contribute 102 of the 116 leaf tasks. Add `belebele_ell_Grek` below and it is 8:6. GreekMMLU additionally has two generative scorings
(`greekmmlu_gen` for 5-shot, `greekmmlu_gen_boxed` for 0-shot) that test whether a log-likelihood change is
calibration or knowledge.

## Greek tasks that already ship with lm-eval

Not built here, but available in the same harness and worth knowing about. Verified present in this fork.

| task | dataset | type | note |
|---|---|---|---|
| `belebele_ell_Grek` | `facebook/belebele` | MCQ | Reading comprehension, **professionally translated** — the lowest-noise MT benchmark in the suite. ILSP reports it 5-shot |
| `global_mmlu_full_el` | `CohereForAI/Global-MMLU` | MCQ | Another Greek MMLU, partly human-verified, with culturally-sensitive vs culturally-agnostic subsets you can report separately. 57 subjects + 4 category groups |
| `include_base_44_greek` | `CohereForAI/include-base-44` | MCQ | **Native Greek exams**, in categories including `medical_license` and `professional_certification` — check for overlap with `ilspgreekmedicalmcqa` and `ilspgreekasep` before reporting both. Three variants: `default`, `few_shot_en`, `few_shot_og` |
| `global_piqa_*_ell_grek` | `mrlbenchmarks/global-piqa-*` | MCQ / generative | Physical commonsense, parallel and non-parallel, cloze and generation |
| `xnli_el` | `facebook/xnli` | MCQ | NLI, professionally translated |
| `xquad_el` | `google/xquad` | generative | Extractive QA, professionally translated |
| `arc_challenge_mt_el` | `LumiOpen/arc_challenge_mt` | MCQ | A second MT of ARC-Challenge — useful as a translation-variance check against `ilspgreekarc_challenge` |
| `multiblimp_ell` / `multiblimp_grc` | `jumelet/multiblimp` | MCQ | Minimal-pair grammaticality. `grc` is **Ancient** Greek |

`mmlu`, `mmlu_pro`, `winogrande`, `mgsm`, `ifeval` (English) are the forgetting guards; `ilspgreekwinogrande`
is the Greek counterpart (see "Not included, and why" for its filter).

## How to run

```bash
lm-eval --model hf --model_args pretrained=<model>,dtype=bfloat16 \
        --tasks greekmmlu --num_fewshot 5 --batch_size 4 --device cuda:0 \
        --log_samples --output_path runs/eval/<name>
```

Always pass `--log_samples`: the generation diagnostic below reads those dumps, and they are the only record of what
the model actually said.

### The two prompt formats

- **Raw few-shot, likelihood** — the setting published Greek numbers use (Meltemi, Llama-Krikri, the Open LLM
  Leaderboard).
- **Chat template** (`--apply_chat_template`, plus `--fewshot_as_multiturn` for few-shot generative tasks) — the
  setting a deployed instruct model actually sees.

**What happens when an instruct model gets a raw prompt: generative tasks read ≈0.** Not because the model is wrong,
but because it emits EOS immediately, the harness records an empty string, and every metric scores that as a failure.
Measured on Meltemi-7B-Instruct-v1.5: IFEval returned empty generations for 8 of 15 rows and scored 0.087
instruction-level, MGSM for 36 of 40 and scored 0.0. With the template: 0.565 and 0.25. A `0.000` on a generative task
is therefore worth checking against the `empty` row of the generation diagnostic before believing it.

### Choosing a protocol: things to weigh

There is no single right answer here, and the trade-offs differ by task type. What is worth thinking about:

**Multiple-choice tasks never generate a token.** Scoring compares the log-likelihood of the supplied options, so an
empty response is impossible and the chat template only changes the surrounding context. Published Greek numbers
(Meltemi, Llama-Krikri, the Open LLM Leaderboard) are raw few-shot likelihood, so raw is the setting that buys
comparability. Applying a template here is defensible if you care about how the model behaves in its deployed form,
but expect it to move the numbers and be explicit that it did.

**Generative tasks are where the choice actually bites.** A model fine-tuned behind a chat template treats a bare
prompt as off-distribution and often emits EOS immediately — the harness records an empty string and every metric
scores it as a failure, so the task reads ~0 for reasons that have nothing to do with ability. The mirror image also
holds: a base checkpoint that never saw a template can be hurt by one. So "run base and fine-tuned through the
identical path" and "run each model in its native format" are in genuine tension for generative tasks, and which one
you want depends on the claim you are making. Reporting both cells and saying which is which costs one extra run and
removes the ambiguity.

**Worth checking first: whether your base model already understands its template.** Some checkpoints do, usually
because mid-training included conversational data — `greekllm.train.probe_chat` in the training repo tests exactly
this. When the base does understand it, the tension above disappears: both sides can take the template, and the path
is identical *and* native.

**An empty response is itself a finding.** Whether a fine-tuned model still answers a bare prompt is part of what
instruction tuning changed, not just an artifact to configure away. The `terminated` and `empty` rows of the
generation diagnostic measure it directly, so a raw-prompt run of the fine-tuned model can be worth keeping as a
data point even when it is not the headline number.

### Few-shot settings

Use the ILSP column in the task table when the point is comparability with their published numbers. `--num_fewshot`
is global, so tasks with different settings need separate invocations.

### Generation diagnostics

MCQ benchmarks rank the log-likelihood of supplied options; the model never generates a token, so they cannot see
whether it stops, answers in Greek, or loops. This reads the sample dumps you already produced:

```bash
python -m greekllm.eval.failure_profile --runs base=runs/eval/base sft=runs/eval/sft \
    --tokenizer <model> [--json profile.json]
```

Reports termination rate, empty rate, language consistency (GlotLID), repetition, mojibake, length sanity, and
IFEval's format-constraint pass rate. It adds no prompts and no labels, so it is a diagnostic, not a benchmark — do
not call it one in a report. Details and calibration in the training repo's `evaluation/TASKS_TODO.md` §5.

## Gotchas

**Reasoning-style chat templates open a think block by default — close it, or every generative score is wrong.**
Qwen3.5 and K2-Horizon templates end the generation prompt with `<think>\n` / `<ifm|think>\n`, so the model reasons
first and the harness cuts or mis-extracts the answer (Qwen3.5-4B-Base read 5% on MGSM). Pass
`enable_thinking=false` for Qwen; K2 has no such switch, so this fork adds the model arg
`chat_template_suffix` (e.g. `"</ifm|think>"`) which is appended verbatim after the rendered generation prompt.
Also pass `--gen_kwargs '{"until": ["<turn-end token>"]}'` for chat runs: the Qwen base snapshot has no
`generation_config.json` and its tokenizer eos is `<|endoftext|>`, so generation would otherwise run to the cap,
and task-level newline stops (MGSM, Civics QA) cut multi-line chat answers after the first line. Note this fork
auto-enables `--fewshot_as_multiturn` with a chat template; K2 needs `--fewshot_as_multiturn false` because its
template rejects assistant messages without a `reasoning_content` field.

**`ilsp/ifeval_greek` stores numeric kwargs as floats** (`num_words: 300.0`); the instruction verifiers index with
them. `ilspgreekifeval/utils.py` coerces whole floats to int before `build_description`, otherwise every row raises
`TypeError` — after all generations have already been paid for.


**ROUGE is silently zero for Greek.** `rouge_score`'s tokenizer replaces every character outside `[a-z0-9]` with a
space, so Greek text tokenizes to an empty list and two identical Greek sentences score 0. `ilspgreektruthfulqa_gen`
ships a Unicode tokenizer to fix this; **anything else in lm-eval that scores Greek generation with ROUGE has the same
hole**. BLEU is unaffected (sacrebleu uses `tokenize="intl"`).

**Numbers are protocol-comparable with ILSP, not bit-identical.** Prompts follow their lighteval fork, but their
implementation's mechanical defects were not reproduced (stray indentation from Python triple-quoted strings, a Latin
`A` in `Aπάντηση`, a truncated primer sentence), and lm-eval scores TruthfulQA mc1/mc2 as separate tasks where
lighteval scores them from one combined choice list. Sanity check on MMLU EL `anatomy`, all 135 questions, 5-shot:
lighteval 0.3556 ± 0.0414 vs this port 0.3926 ± 0.0422.

**Greek letters are not Latin letters.** `ilspgreekmmlu` and `greekmmlu` both score Greek `Α/Β/Γ/Δ` (U+0391–0394), not
Latin `A/B/C/D` — different tokens with different priors. If you add an MCQ task, match the existing convention or the
comparison between them is contaminated by the label choice.

**`ilspgreekmedicalmcqa` evaluates on `train`.** That is what ILSP's own config does, despite the split names, and it
is what their published 48.0% refers to. Swap `test_split` and `fewshot_split` in the yaml if you prefer the held-out
half — and then stop comparing against their number.

**Long few-shot contexts OOM at large batch sizes.** 5-shot MMLU at `--batch_size 16` on a 4B model tried to allocate
37 GiB of logits on a 48 GiB card. Use 4, or `--batch_size auto`.

**Only `ilspgreekifeval` can measure termination**, because it is the one task configured with `until: []`. Everywhere
else lm-eval cuts the generation at a stop string.

**Benchmark data must never enter training.** Re-run the eval blacklist over any synthetic data generated by a strong
model — it can reproduce benchmark items verbatim.

## Not included, and why

- **`ilsp/winogrande_greek`** — *now included as `ilspgreekwinogrande`, filtered* (2026-09-21). The dataset keeps
  `sentence`, `option1` and `option2` in **English**; only `multiple_choice_targets` is Greek, and its two candidates
  are independently machine-translated. Measured over the full validation split: 18.9% of pairs have the whole
  sentence retranslated differently and 0.5% leave a Latin name in one branch only. Both classes are dropped in
  `process_docs` (1,021 of 1,267 rows survive; the few-shot pool is filtered the same way) and the surviving pair is
  scored upstream-`winogrande` style: shared prefix as context, only the differing remainders compared. ILSP's own
  suite has no winogrande task; scores are not comparable to anything computed on the unfiltered pairs.
- **MT-Bench Greek, Arena-Hard Greek** — multi-turn and pairwise judging do not fit lm-eval. The Greek judge prompts
  are saved in the training repo's `evaluation/ilsp-assets/`.
- **`ilsp/greek_lyceum_mathematics`** — answers are worked solutions, so it needs a scoring decision (judge, or
  extraction from the final line) before it can be a task.
- **ILSP's lighteval fork as a runner** — tested and rejected. It pins `torch<2.5`, which forces transformers down to
  4.46 and makes it unable to load Qwen3.5-4B or K2 at all, and it needed six separate fixes before it would run once.
  Its Greek IFEval instruction library was worth taking; the rest was a ~20-line prompt function per task.
