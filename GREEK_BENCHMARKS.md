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

Each model is evaluated as **17 groups**: one lm-eval call per group, one shot count per call, one output folder per
group (see [Output format](#output-format)). The group names are those of the toolkit's driver
(`python -m greekeval.run_suite --groups ...`). Always name the groups to run with `--groups`: the
driver's default plan leaves out the five form-B groups.

| group | form | shots | tasks | items run | max new tokens | prompt |
|---|---|---:|---|---|---|---|
| `mcq_fs0` | A | 0 | `greekmmlu`, `ilspgreektruthfulqa_mc1`, `ilspgreektruthfulqa_mc2` | 16,632 · 817 · 817 | — | raw |
| `mcq_fs5` | A | 5 | `greekmmlu`, `belebele_ell_Grek`, `ilspgreekasep` | 16,632 · 900 · 2,346 | — | raw |
| `mcq_fs5b` | A | 5 | `ilspgreekmmlu` | 14,042 | — | raw |
| `mcq_fs5w` | A | 5 | `ilspgreekwinogrande` | 1,021 | — | raw |
| `mcq_fs5_en` | A | 5 | `mmlu` | 14,042 | — | raw |
| `mcq_fs10` | A | 10 | `ilspgreekhellaswag` | 10,024 | — | raw |
| `mcq_fs15` | A | 15 | `ilspgreekmedicalmcqa` | 1,602 | — | raw |
| `mcq_fs25` | A | 25 | `ilspgreekarc_easy`, `ilspgreekarc_challenge` | 2,376 · 1,168 | — | raw |
| `gen_mmlu_fs0` | B | 0 | `greekmmlu_gen_boxed` | 16,632 | 64 (1,024 with the chat template) | raw |
| `gen_mmlu_fs5` | B | 5 | `greekmmlu_gen` | 16,632 | 8 | raw |
| `gen_ilsp_fs5` | B | 5 | `ilspgreekmmlu_gen` | 14,042 | 8 | raw |
| `gen_bele_fs5` | B | 5 | `belebele_ell_Grek_gen` | 900 | 8 | raw |
| `gen_mmlu_en_fs5` | B | 5 | `mmlu_generative` | 14,042 | 256 | raw |
| `gen_fs0` | C | 0 | `ilspgreekifeval`, `ifeval` | 541 · 541 | 1,280 | chat |
| `gen_civics` | C | 0 | `ilspgreekcivicsqa` | 407 | 100 | chat |
| `gen_trans` | C | 0 | `ilspgreekflores_en_el`, `ilspgreekflores_el_en`, `ilspgreektruthfulqa_gen` | 1,012 · 1,012 · 817 | 100 · 100 · 256 | chat |
| `gen_fs8b` | D | 8 | `ilspgreekmgsm` | 250 | **2,048** (must be set) | chat |

- **Items run: the full test set, always.** No `--limit` in a campaign run. The `limit` and `n_samples` columns of
  `all_scores.csv` tell a capped row from a full one; never put the two in one comparison: the first items of a test
  set are not a random sample.
- **Only the test split is scored.** The dev split supplies the few-shot exemplars and nothing else: GreekMMLU is
  16,632 test items, and its 225 dev items are never counted.
- **Letter groups with the chat template** (`_chat`): a chat-tuned model explains before it answers, so the 64-token
  cap of `gen_mmlu_fs0` cuts it off before the box; a model that answers directly needs a handful of tokens. So
  `gen_mmlu_fs0` takes `max_gen_toks` 1,024 whenever it runs with the chat template (the driver sets it). The 5-shot
  groups keep 8 tokens: multi-turn exemplars make the model answer with the letter. A model whose template cannot
  hold multi-turn exemplars gets them inside one user turn and may write an explanation instead: read `parsed`
  before the score.
- **Chat prompt** = each model's own template (`--apply_chat_template`) with any think block closed. A base model that
  cannot follow its template runs sets C and D as raw completion instead, into `<group>_raw/`.
- **0-shot log-likelihood** (`greekmmlu` in `mcq_fs0`) can underrate a chat-tuned model: fine-tuning shifts the
  calibration of single-letter continuations. Quote the 5-shot row, or check it with `gen_mmlu_fs0`.
- **Retired:** `gen_fs8` is MGSM at lm-eval's default 256-token cap, which cuts Greek chain-of-thought answers off.
  Never use it; `gen_fs8b` replaces it.

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
scored and stored as the answer. Driver: `run_suite --think
--out_root runs/eval_think` for think on in forms C and D. Form B needs `--chat_mcq` as well, in a call of its own
(`run_suite --think --chat_mcq --groups gen_mmlu_fs0 --out_root runs/eval_think`); with `--think` alone the driver
skips the letter groups. `run_suite --chat_mcq --suffix _closed --out_root runs/eval_think` for a B control. A fourth
state, *the model decides* (no think tag after the assistant header), is not a benchmark setting.

Two things are left alone on purpose in think-on runs. **Decoding is greedy**, like the rest of the campaign, not the
sampling Qwen's model card recommends for thinking (temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5).
**The language of the reasoning is the model's own choice**: nothing in the prompt sets it.

Report, for think on: the share of answers that never closed the block within the cap, and the gain on the
questions that did finish, besides the overall score. Compare with think off on the same items. An answer counts as
never closed when its stored response, re-tokenised, is within 16 tokens of the cap (≥ 3,056 of 3,072): a closed
block leaves only the answer in the sample. The rule was checked against runs that kept the raw output.

Where a think-on run and its think-off control go:

| form | think on (`runs/eval_think/<model>/`) | think-off control |
|---|---|---|
| B | `gen_mmlu_fs0_chat` (GreekMMLU 0-shot) | `runs/eval_think/<model>/gen_mmlu_fs0_chat_closed` |
| C | `gen_fs0` (IFEval el + en) | `runs/eval/<model>/gen_fs0` |
| D | `gen_fs8b` (MGSM) | `runs/eval/<model>/gen_fs8b` |

For B, prefer 0-shot: 5-shot exemplars show direct answers, which invite the model to skip thinking.

## Output format

Every run writes to this layout — the collector, the score export and the status tool read nothing else:

```
runs/eval/<model>/<group><suffix>/<sanitized model path>/
    results_<timestamp>.json              # exactly one per folder
    samples_<task>_<timestamp>.jsonl      # one per task or subject (--log_samples)
```

| part | rule | examples |
|---|---|---|
| `<model>` | short lowercase key `<family>_<stage>`; stage = `base`, `instruct` (a published chat model) or a name for a fine-tuning run. Register the key and its local path in the project's model registry (`configs/eval/models.yaml`, see [`README.md`](README.md)); the path never goes in the folder name | `mymodel_base`, `mymodel_sft1`, `other_instruct` |
| `<group>` | a group name from the table above, unchanged. A new task needs a new group in `run_suite.GROUPS` first | `mcq_fs5`, `gen_fs8b` |
| `<suffix>` | the protocol variant; none = the group's own protocol from the table | `_raw`: a chat group run as raw completion (base models) · `_chat`: a letter group run with the chat template · `_closed`: the think-off control of a think-on run, only under `runs/eval_think/` (B groups: `_chat_closed`) |
| `<sanitized model path>` | created by lm-eval from `pretrained` (`/` → `__`); do not create or rename it | `__path__to__models__mymodel_sft1` |

Rules:

1. **One lm-eval call = one group = one folder.** Do not put tasks from two groups in one call: the collector keys
   results by folder, so a mixed call lands in the wrong place.
2. **Exactly one `results_*.json` per folder.** To re-run, delete the folder first. The collector reads every results
   file it finds, so a second one silently mixes old and new numbers. `run_suite` skips folders that already have
   results.
3. **Always** `--log_samples --seed 1234`, `"dtype": "bfloat16"`, greedy decoding (the task defaults). The samples are
   the only record of what the model wrote; subset re-scoring and the failure profile read them.
4. **No `--limit`.** Timing probes or other capped runs go to a scratch root, never under `runs/eval/`.
5. **Set generation caps explicitly** as in the table, including under `_raw`: `--gen_kwargs '{"max_gen_toks": 2048}'`
   for MGSM. With the chat template, also stop at the model's turn-end token (`"until": ["<|im_end|>"]` for Qwen).
6. **Every think-open run MUST go to `runs/eval_think/`, never `runs/eval/`.** Group names are shared between the two
   roots, so the root is what tells a think-open result from a think-closed one. `run_suite --think` refuses
   `runs/eval`; with plain lm-eval set `ROOT=runs/eval_think` (see [Commands](#commands)). Other diagnostics (probes,
   non-standard limits) also stay out of `runs/eval/`.
7. Write nothing else inside `runs/eval/<model>/`. The tables at the top of `runs/eval/` (`summary*.csv/md`,
   `STATUS.md`, `profile_*.json`) are generated. After a run,
   `python -m greekeval.export_scores` rewrites `reports/eval_results/all_scores.csv` from these folders.

The simplest way to comply is the driver, which builds every path and flag itself:

```bash
python -m greekeval.run_suite --gpu 0 --models <key> --groups mcq_fs5 gen_fs8b   # --raw_gen / --chat_mcq for variants
python -m greekeval.run_suite --dp 2 --models <key> --groups gen_mmlu_fs0        # one run over both GPUs
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

Without the driver, run each group as one call with the flags below; `run` writes the folder layout of
[Output format](#output-format). Keep the batch sizes (they fit a 4B model on a 48 GB card; halve them for larger models).

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
MODEL=mymodel_base                                                   # the <model> key
M='{"pretrained": "/path/to/model", "dtype": "bfloat16"}'            # + "trust_remote_code": true if the model needs it
run() {   # run <group> <tasks> <shots> <batch> [extra lm-eval args...]
  local g=$1 t=$2 n=$3 b=$4; shift 4
  lm-eval --model hf --model_args "$M" --tasks "$t" --num_fewshot "$n" --batch_size "$b" --device cuda:0 \
          --log_samples --seed 1234 --output_path "${ROOT:-runs/eval}/$MODEL/$g" "$@"
}                                                                    # ROOT=runs/eval_think for think-open runs
```

### A. Log-likelihood (raw prompt, every model the same way)

```bash
run mcq_fs0    greekmmlu,ilspgreektruthfulqa_mc1,ilspgreektruthfulqa_mc2 0  8
run mcq_fs5    greekmmlu,belebele_ell_Grek,ilspgreekasep                 5  4
run mcq_fs5b   ilspgreekmmlu                                             5  4
run mcq_fs5w   ilspgreekwinogrande                                       5  8
run mcq_fs5_en mmlu                                                      5  4
run mcq_fs10   ilspgreekhellaswag                                        10 4
run mcq_fs15   ilspgreekmedicalmcqa                                      15 2
run mcq_fs25   ilspgreekarc_easy,ilspgreekarc_challenge                  25 2
```

### B. Letter-writing (raw prompt, same exemplars as set A)

```bash
run gen_mmlu_fs0    greekmmlu_gen_boxed   0 8
run gen_mmlu_fs5    greekmmlu_gen         5 8
run gen_ilsp_fs5    ilspgreekmmlu_gen     5 8
run gen_bele_fs5    belebele_ell_Grek_gen 5 8
run gen_mmlu_en_fs5 mmlu_generative       5 8
```

The `_chat` variant (e.g. `gen_mmlu_fs5_chat`) adds the chat flags below; `gen_mmlu_fs0_chat` also sets
`"max_gen_toks": 1024`. Compare each group with the set-A row at the
**same shot count**: `gen_mmlu_fs5` ↔ `greekmmlu` in `mcq_fs5`, `gen_mmlu_fs0` ↔ `greekmmlu` in `mcq_fs0`. Read
`exact_match` together with `parsed`.

### C and D. Open-ended and reasoning (chat template)

```bash
T='"until": ["<|im_end|>"]'                                          # the model's turn-end token
run gen_fs0    ilspgreekifeval,ifeval 0 8 --apply_chat_template --gen_kwargs "{$T}"
run gen_civics ilspgreekcivicsqa      0 8 --apply_chat_template --gen_kwargs "{$T, \"max_gen_toks\": 100}"
run gen_trans  ilspgreekflores_en_el,ilspgreekflores_el_en,ilspgreektruthfulqa_gen 0 8 --apply_chat_template --gen_kwargs "{$T}"
run gen_fs8b   ilspgreekmgsm          8 8 --apply_chat_template --gen_kwargs "{$T, \"max_gen_toks\": 2048}" --fewshot_as_multiturn true
```

Reasoning-style templates open a think block by default — close it or every generative score is wrong: Qwen3.5 takes
`"enable_thinking": false` in the model args; K2-Horizon takes `"chat_template_suffix": "</ifm|think>"` (the fork's
addition) and `--fewshot_as_multiturn false`.

Think block open (a Qwen-style template shown) — separate root, 3,072-token cap, model args
`"enable_thinking": true, "think_end_token": "</think>"`:

```bash
ROOT=runs/eval_think run gen_fs8b ilspgreekmgsm 8 8 --apply_chat_template --gen_kwargs "{$T, \"max_gen_toks\": 3072}" --fewshot_as_multiturn true
```

For a base model that cannot follow its template, drop `--apply_chat_template` and the `until`, keep the caps, and
write to `<group>_raw`:

```bash
run gen_fs8b_raw ilspgreekmgsm 8 8 --gen_kwargs '{"max_gen_toks": 2048}'
```

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
