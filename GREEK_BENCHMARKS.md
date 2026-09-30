# Greek benchmarks — quick reference

One page: which tasks exist, how each is run and scored, and where the output must go so that runs made by different
people or agents land in one comparable table. Install and tool commands are in [`README.md`](README.md);
background, protocol trade-offs and every known pitfall are in [`GREEK_EVAL.md`](GREEK_EVAL.md).

## At a glance

**24 tasks** are run (21 Greek, 3 English guards), in four forms. A *task* is one lm-eval task name; `greekmmlu`,
`ilspgreekmmlu` and `mmlu` aggregate their subjects and count once. `greekmmlu` runs at 0 and 5 shots, so the 24 tasks
make 25 task × shot settings: 10 at 0-shot, 10 at 5-shot, 5 at other counts.

| form | what the model does | tasks, by shots | # |
|---|---|---|---:|
| **A. Log-likelihood** (multiple choice) | generates nothing; the harness ranks the options. Raw prompt for every model — what published Greek numbers use | **0:** `greekmmlu`, `ilspgreektruthfulqa_mc1`, `ilspgreektruthfulqa_mc2` · **5:** `greekmmlu`, `belebele_ell_Grek`, `ilspgreekasep`, `ilspgreekmmlu`, `ilspgreekwinogrande`, `mmlu` (en) · **10:** `ilspgreekhellaswag` · **15:** `ilspgreekmedicalmcqa` · **25:** `ilspgreekarc_easy`, `ilspgreekarc_challenge` | 12 |
| **B. Generative, letter-writing** | writes the option letter of a set-A question; checks set A for calibration artefacts | **0:** `greekmmlu_gen_boxed` · **5:** `greekmmlu_gen`, `ilspgreekmmlu_gen`, `belebele_ell_Grek_gen`, `mmlu_generative` (en) | 5 |
| **C. Generative, open-ended** | writes free text, scored by rule checkers or against references | **0:** `ilspgreekifeval`, `ifeval` (en), `ilspgreekcivicsqa`, `ilspgreekflores_en_el`, `ilspgreekflores_el_en`, `ilspgreektruthfulqa_gen` | 6 |
| **D. Reasoning** (chain of thought) | writes a worked solution, scored on the final answer | **8:** `ilspgreekmgsm` (run for every model) · *not run:* `ilspgreekmmlupro`, 0-shot, implemented but skipped (12K long generations) | 1 |

Which runs to make is not decided here but in the plan file, `config/plan.yaml`: one line per run, with its form, tasks, shots, prompt, cap and output folder, and
comments explaining the choices. The driver runs every line that is not commented out. Read that file for the
inventory; this page only states what a run must obey.

- **Only the test split is scored.** The dev split supplies the few-shot exemplars and nothing else: GreekMMLU is
  16,632 test items, and its 225 dev items are never counted.
- **Items run: the full test set.** The `limit` and `n_samples` columns of `all_scores.csv` tell a capped row from a
  full one; never put the two in one comparison: the first items of a test set are not a random sample.
- **Chat prompt** = each model's own template (`--apply_chat_template`) with any think block closed. A base model that
  cannot follow its template runs the C and D lines as raw completion instead (the `_raw` lines).
- **0-shot log-likelihood** (`greekmmlu` at 0 shots) can underrate a chat-tuned model: fine-tuning shifts the
  calibration of single-letter continuations. Quote the 5-shot row, or check it with the 0-shot letter-writing run.
- **Read `parsed` next to a letter-writing score.** An unparsed answer counts as wrong; a model whose template
  cannot hold multi-turn exemplars may explain instead of answering with the letter.

### Thinking settings — a protocol variant of B, C and D

Any task the model writes an answer for can run under three settings; set A cannot, since it generates nothing.
Thinking needs the chat template (the think block lives there) and a model that has a think mode. For a model
without one, or one fine-tuned to close the block at once, only the first two rows exist.

| setting | prompt | think block | model args | where it goes |
|---|---|---|---|---|
| **no template** — the default for a base model | raw | none | — | `runs/eval/<model>/<group>` for B; `<group>_raw` for C and D |
| **think off** (forced) — the campaign protocol | chat | closed, empty | Qwen `"enable_thinking": false`; K2 `"chat_template_suffix": "</ifm\|think>"` | `runs/eval/<model>/<group>` for C and D; `<group>_chat` for B. Exception: the control run next to a think-on run goes to `runs/eval_think/<model>/<group>_closed` (B: `<group>_chat_closed`) |
| **think on** (forced) | chat | open, 3,072 new tokens for reasoning + answer | Qwen `"enable_thinking": true, "think_end_token": "</think>"`; K2 no suffix, `"think_end_token": "</ifm\|think>"` | **must** go to `runs/eval_think/<model>/<group>`, never `runs/eval/` — the group name is the same as think-off, only the root tells them apart |

lm-eval strips the reasoning (everything up to `think_end_token`) before it scores and before it writes the samples,
so a sample holds the answer only. When the block never closed there is nothing to strip: the reasoning itself is
scored and stored as the answer. In the plan, think on is the block of lines with `root: eval_think, think: on`
(one per C, D and B run) and the B controls are the `_chat_closed` lines next to them; comment out what you do not
want. A model whose registry entry has no `think` field is skipped on those lines: it has no block to open. A fourth
state, *the model decides* (no think tag after the assistant header), is not a benchmark setting.

Two things are left alone on purpose in think-on runs. **Decoding is greedy**, like the rest of the campaign, not the
sampling Qwen's model card recommends for thinking (temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5).
**The language of the reasoning is the model's own choice**: nothing in the prompt sets it.

Report, for think on: the share of answers that never closed the block within the cap, and the gain on the
questions that did finish, besides the overall score. Compare with think off on the same items. An answer counts as
never closed when its stored response, re-tokenised, is within 16 tokens of the cap (≥ 3,056 of 3,072): a closed
block leaves only the answer in the sample. The rule was checked against runs that kept the raw output.

For B prefer 0-shot: 5-shot exemplars show direct answers, which invite the model to skip thinking.

## Output format

Every run writes to this layout — the collector, the score export and the status tool read nothing else:

```
runs/eval/<model>/<group><suffix>/<sanitized model path>/
    results_<timestamp>.json              # exactly one per folder
    samples_<task>_<timestamp>.jsonl      # one per task or subject (--log_samples)
```

| part | rule | examples |
|---|---|---|
| `<model>` | short lowercase key `<family>_<stage>`; stage = `base`, `instruct` (a published chat model) or a name for a fine-tuning run. Register the key and its local path in the project's model registry (`config/models.yaml`, see [`README.md`](README.md)); the path never goes in the folder name. The key is the shared name: one model carries the same key on every machine, only its path in the registry differs | `mymodel_base`, `mymodel_sft1`, `other_instruct` |
| `<group>` | a run name from the plan, unchanged. A new task needs a new line in the plan first | `mcq_fs5`, `gen_fs8b` |
| `<suffix>` | the protocol variant; none = the group's own protocol as the plan defines it | `_raw`: a chat group run as raw completion (base models) · `_chat`: a letter group run with the chat template · `_closed`: the think-off control of a think-on run, only under `runs/eval_think/` (B groups: `_chat_closed`) |
| `<sanitized model path>` | created by lm-eval from the `pretrained` string (`/` → `__`); do not create or rename it. It follows where the model sits, so it differs from machine to machine, and the tools never read its name. With plain lm-eval pass the absolute path without a trailing slash: another spelling of the same path makes another folder | `__path__to__models__mymodel_sft1` |

Rules:

1. **One lm-eval call = one group = one folder.** Do not put tasks from two groups in one call: the collector keys
   results by folder, so a mixed call lands in the wrong place.
2. **Exactly one `results_*.json` per folder.** To re-run, delete the folder first. The collector reads every results
   file it finds, so a second one silently mixes old and new numbers. `run_suite` skips folders that already have
   results (also under `--limit`), and `status` flags a folder with two as `DUP`. The same holds for results brought from another machine: their subfolder has another name, so a copy
   into a folder that already has results leaves two side by side and the tools read one of them without saying
   which. One `<model>/<group>` folder comes from one machine; copy it whole.
3. **Always** `--log_samples --seed 1234`, `"dtype": "bfloat16"`, greedy decoding (the task defaults). The samples are
   the only record of what the model wrote; subset re-scoring and the failure profile read them.
4. **No `--limit` in a campaign run.** A smoke test is `run_suite --limit 1 --runs runs/smoke`: the same layout under
   `runs/smoke/eval` and `runs/smoke/eval_think`, 1 item per task or subject; the driver refuses `--limit` into `runs/`.
   Read such a folder with `collect --root runs/smoke/eval`, `status --runs runs/smoke` and
   `export_scores --runs runs/smoke --out <file>`. Expect a few items to score differently from a full run: bf16
   ties flip with the batch composition.
5. **Set generation caps explicitly** as in the table, including under `_raw`: `--gen_kwargs '{"max_gen_toks": 2048}'`
   for MGSM. With the chat template, also stop at the model's turn-end token (`"until": ["<|im_end|>"]` for Qwen).
6. **Every think-open run MUST go to `runs/eval_think/`, never `runs/eval/`.** Group names are shared between the two
   roots, so the root is what tells a think-open result from a think-closed one. The plan keeps every think-on line
   under `root: eval_think`; with plain lm-eval write to `runs/eval_think/<model>/<group>` yourself. Other
   diagnostics (probes, non-standard limits) also stay out of `runs/eval/`.
7. Write nothing else inside `runs/eval/<model>/`. The tables at the top of `runs/eval/` (`summary*.csv/md`,
   `STATUS.md`, `profile_*.json`) are generated. After a run,
   `python -m greekeval.export_scores` rewrites `reports/eval_results/all_scores.csv` from these folders.

Across machines, three things must be identical for two runs to land in the same cell of the table: the `<model>`
key, the `<group><suffix>` name and the root (`runs/eval` or `runs/eval_think`). The sanitized path and the
timestamps may differ. The layout says where a number goes, not that two machines get the same number: keep rule 3,
the caps and the batch sizes, and compare `n_samples` and `limit` in `all_scores.csv` before comparing scores.

The simplest way to comply is the driver, which builds every path and flag from the plan:

```bash
python -m greekeval.run_suite --list --models <key>                              # every line of the plan: done or todo
python -m greekeval.run_suite --gpu 0 --models <key>                             # run them all (skips what is done)
python -m greekeval.run_suite --gpu 0 --models <key> --groups mcq_fs5 gen_fs8b   # only these lines
python -m greekeval.run_suite --dp 2 --models <key> --groups gen_mmlu_fs0        # one run over both GPUs
python -m greekeval.run_suite --gpu 0 --models <key> --limit 1 --runs runs/smoke # smoke test
```

## Setup

Install: see [`README.md`](README.md). Datasets download from the Hub on first use; afterwards everything runs with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

## Tasks

*type*: **MCQ** = the harness ranks the log-likelihood of the candidates, the model generates nothing; **gen** = the model
generates text that is then scored. *prompt*: **raw** = plain few-shot completion (what published Greek numbers use);
**chat** = `--apply_chat_template` (needed for instruct models on gen tasks — without it they emit EOS and score ≈0).
*shots* = the setting used in this project (ILSP's protocol where they publish one).

### Built in the fork

| task | measures | type | scored on | result key | items | shots | prompt | source |
|---|---|---|---|---|---:|---:|---|---|
| `greekmmlu` | knowledge, 45 subjects | MCQ | option letter Α/Β/Γ/Δ | `acc,none` | 16,632 | 5 (and 0) | raw | native |
| `greekmmlu_gen` | same, model **writes** the letter | gen | first standalone letter | `exact_match,letter` (+`parsed`) | 16,632 | 5 | raw | native |
| `greekmmlu_gen_boxed` | same, `\boxed{}` instruction for 0-shot | gen | letter inside `\boxed{}` | `exact_match,boxed` (+`parsed`) | 16,632 | 0 | raw | native |
| `ilspgreekmmlu` | MMLU translated, 57 subjects | MCQ | option letter | `acc,none` | 14,042 | 5 | raw | MT |
| `ilspgreekmmlu_gen` | same, model writes the letter | gen | first standalone letter | `exact_match,letter` (+`parsed`) | 14,042 | 5 | raw | MT |
| `ilspgreekmmlupro` | MMLU-Pro translated, 10-way, chain-of-thought | gen | extracted final letter | `exact_match,none` | 12,032 | 0 | chat for instruct | MT |
| `ilspgreekarc_easy` / `_challenge` | science QA | MCQ | option text | `acc`, `acc_norm` | 2,376 / 1,168 | 25 | raw | MT |
| `ilspgreekhellaswag` | commonsense sentence completion | MCQ | ending text | `acc`, `acc_norm` | 10,024 | 10 | raw | MT |
| `ilspgreektruthfulqa_mc1` / `_mc2` | truthfulness | MCQ | statement text | `acc,none` | 817 | 0 | raw | MT, human-checked |
| `ilspgreektruthfulqa_gen` | truthfulness, free answer | gen | BLEU/ROUGE vs true and false references | `bleu_acc`, `rouge1_acc`, … | 817 | 0 | chat for instruct | MT, human-checked |
| `ilspgreekmedicalmcqa` | medical licensing exams, 5-way | MCQ | option text | `acc`, `acc_norm` | 1,602 | 15 | raw | native |
| `ilspgreekasep` | civil-service exams | MCQ | option text | `acc`, `acc_norm` | 2,346 | 5 | raw | native |
| `ilspgreekwinogrande` | pronoun resolution, filtered pairs | MCQ | shared continuation (winogrande-style) | `acc`, `acc_norm` | 1,021 | 5 | raw | MT, filtered |
| `ilspgreekifeval` | verifiable instruction following | gen | 27 rule checkers (Greek-aware) | `prompt_level_strict_acc`, `inst_level_strict_acc` (+ `_loose_`) | 541 | 0 | chat | translated + localised |
| `ilspgreekmgsm` | grade-school math, chain-of-thought | gen | final number | `exact_match,flexible-extract` / `strict-match` | 250 | 8 | chat | MT |
| `ilspgreekcivicsqa` | civics, free-form answer | gen | corpus BLEU vs textbook answer | `bleu,none` | 407 | 0 | chat | native |
| `ilspgreekflores_en_el` / `_el_en` | translation en↔el | gen | BLEU, chrF | `bleu,none`, `chrf,none` | 1,012 each | 0 | chat | professional |
| `belebele_ell_Grek_gen` | reading comprehension, model writes the letter | gen | first standalone letter | `exact_match,letter` (+`parsed`) | 900 | 5 | raw | professional |

Groups: `greekmmlu`, `greekmmlu_gen`, `greekmmlu_gen_boxed`, `ilspgreekmmlu`, `ilspgreekmmlu_gen` each aggregate their
subjects (GreekMMLU also has `_stem` / `_humanities` / `_social_sciences` / `_other`). Tags `ilspgreekarc`,
`ilspgreektruthfulqa`, `ilspgreekflores` run their members together.

### Shipped with upstream lm-eval, used here as-is

| task | measures | type | result key | items | shots | prompt | source |
|---|---|---|---|---|---:|---:|---|
| `belebele_ell_Grek` | reading comprehension | MCQ | `acc`, `acc_norm` | 900 | 5 | raw | professional |
| `mmlu` | English knowledge (forgetting guard) | MCQ | `acc,none` | 14,042 | 5 | raw | English |
| `mmlu_generative` | same, model writes the letter (whole first line must be the letter) | gen | `exact_match,get_response` | 14,042 | 5 | raw | English |
| `ifeval` | English instruction following (guard) | gen | `inst_level_strict_acc`, … | 541 | 0 | chat | English |

Available but not part of the 24: `global_mmlu_full_el` (another Greek MMLU), `include_base_44_greek` (native Greek
exams, may overlap medical/ASEP), `xnli_el` / `xquad_el`, `arc_challenge_mt_el` (a second ARC translation),
`global_piqa_*_ell_grek`, `multiblimp_ell` / `_grc`.

## Commands

The driver builds every lm-eval call from the plan. To run one line by hand (from the project root: the output path
is relative to the current directory), translate its fields:

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
M='{"pretrained": "/abs/path/to/model", "dtype": "bfloat16"}'   # + "trust_remote_code": true if the model needs it
lm-eval --model hf --model_args "$M" --tasks <tasks> --num_fewshot <shots> --batch_size <batch> --device cuda:0 \
        --log_samples --seed 1234 --output_path runs/<root>/<model key>/<name>
```

- `template: chat` adds `--apply_chat_template --gen_kwargs '{"until": ["<turn-end token>"]}'` (plus
  `"max_gen_toks": <cap>` when the line has one), and `--fewshot_as_multiturn true` for shots > 0. Reasoning-style
  templates open a think block by default — close it, or every generative score is wrong: a Qwen-style template takes
  `"enable_thinking": false` in the model args; a K2-style one `"chat_template_suffix": "</ifm|think>"` (the fork's
  addition) and `--fewshot_as_multiturn false`, since its template cannot hold multi-turn exemplars.
- `think: on` instead leaves the block open: Qwen-style `"enable_thinking": true, "think_end_token": "</think>"`,
  K2-style no suffix and `"think_end_token": "</ifm|think>"`; `root: eval_think`.
- `template: none` with a cap adds `--gen_kwargs '{"max_gen_toks": <cap>}'` and nothing else: the task's own stop
  rules apply.

### Generation diagnostic

```bash
python -m greekeval.failure_profile --runs base=runs/eval/<key> sft=runs/eval/<key> --tokenizer /path/to/model
```

Reads the `--log_samples` dumps; reports termination, empty replies, language (GlotLID), repetition, and IFEval's
format pass rate. It is a diagnostic, not a benchmark.

## Reading the numbers

- A generative `0.000` almost always means empty replies, not a wrong model: check `parsed` (letter tasks) or the
  diagnostic's `empty` row before believing it.
- Report MCQ numbers as raw few-shot log-likelihood when comparing with Meltemi / Llama-Krikri / the Open LLM
  Leaderboard; that is the protocol they use.
- `ilspgreekmedicalmcqa` evaluates on the `train` split — that is ILSP's own configuration and what their published number refers to.
- Machine-translated benchmarks (ILSP MMLU, ARC, HellaSwag, MGSM, WinoGrande) carry translation noise and contamination
  risk; the native ones (GreekMMLU, medical, ASEP, civics) and the professionally translated ones (Belebele, FLORES)
  are the ones to weight.
