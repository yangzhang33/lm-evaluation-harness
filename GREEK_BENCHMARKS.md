# Greek benchmarks — quick reference

One page: which tasks exist, what each one measures and how it is scored, and the commands to run them.
Background, protocol trade-offs and every known pitfall are in [`GREEK_EVAL.md`](GREEK_EVAL.md).

## Setup

```bash
pip install -e .                                       # this fork of lm-evaluation-harness
pip install langdetect immutabledict                   # ilspgreekifeval
python -c "import nltk; nltk.download('punkt_tab')"    # Greek sentence splitting, ~11 MB
```

Datasets download from the Hub on first use; afterwards everything runs with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`.

## Tasks

*type*: **MCQ** = the harness ranks the log-likelihood of the candidates, the model generates nothing; **gen** = the model
generates text that is then scored. *prompt*: **raw** = plain few-shot completion (what published Greek numbers use);
**chat** = `--apply_chat_template` (needed for instruct models on gen tasks — without it they emit EOS and score ≈0).
*shots* = the setting used in this project (ILSP's protocol where they publish one).

### Built in this fork

| task | measures | type | scored on | result key | items | shots | prompt | source |
|---|---|---|---|---|---:|---:|---|---|
| `greekmmlu` | knowledge, 45 subjects | MCQ | option letter Α/Β/Γ/Δ | `acc,none` | 16,857 | 5 (and 0) | raw | native |
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
| `global_mmlu_full_el` | another Greek MMLU, culturally-sensitive/agnostic subsets | MCQ | `acc,none` | 14,042 | 5 | raw | MT, partly verified |
| `include_base_44_greek` | native Greek exams (may overlap medical/ASEP) | MCQ | `acc,none` | — | 0 / 5 | raw | native |
| `xnli_el` / `xquad_el` | NLI / extractive QA | MCQ / gen | `acc` / `exact_match`, `f1` | 5,010 / 1,190 | 0 | raw | professional |
| `arc_challenge_mt_el` | a second translation of ARC-Challenge | MCQ | `acc`, `acc_norm` | 1,172 | 25 | raw | MT |
| `global_piqa_*_ell_grek`, `multiblimp_ell` / `_grc` | physical commonsense / grammaticality (`grc` = Ancient Greek) | MCQ | `acc` | — | 0 | raw | — |

## Which set to run

Three self-contained sets. Pick by the question you are asking; each line is one `lm-eval` call (template under
*Commands* below), and the `run_suite --groups` names are for the training repo's driver.

### A. Log-likelihood only (multiple choice) — the headline set, comparable with published Greek numbers

Raw prompt for every model, no chat template. Nothing is generated, so there is no parse column to watch.

| shots | tasks | `run_suite --groups` |
|---:|---|---|
| 5 | `greekmmlu`, `belebele_ell_Grek`, `ilspgreekasep` | `mcq_fs5` |
| 5 | `ilspgreekmmlu` | `mcq_fs5b` |
| 5 | `ilspgreekwinogrande` | `mcq_fs5w` |
| 5 | `mmlu` (English guard, `--limit 40`) | `mcq_fs5_en` |
| 0 | `greekmmlu`, `ilspgreektruthfulqa_mc1`, `ilspgreektruthfulqa_mc2` | `mcq_fs0` |
| 10 | `ilspgreekhellaswag` | `mcq_fs10` |
| 15 | `ilspgreekmedicalmcqa` | `mcq_fs15` |
| 25 | `ilspgreekarc_easy`, `ilspgreekarc_challenge` | `mcq_fs25` |

Caveat: the 0-shot `greekmmlu` line underrates chat-tuned models by 2–4 pt (letter calibration); use set B/C to check
it, or quote the 5-shot number.

### B. Generative, 5-shot — the model writes the letter; raw prompt, same exemplars as set A

Use it to verify a log-likelihood result: at 5-shot it agrees with set A within 0.7 pt on every model tested, so a
disagreement means a calibration artefact, not knowledge. Read `exact_match` together with `parsed`.

| shots | tasks | `run_suite --groups` |
|---:|---|---|
| 5 | `greekmmlu_gen` | `gen_mmlu_fs5` |
| 5 | `ilspgreekmmlu_gen` | `gen_ilsp_fs5` |
| 5 | `belebele_ell_Grek_gen` | `gen_bele_fs5` |
| 5 | `mmlu_generative` (English guard, `--limit 40`; no `parsed` key) | `gen_mmlu_en_fs5` |
| 8 | `ilspgreekmgsm` — the one few-shot generative task that needs the chat template for instruct models | `gen_fs8` |

### C. Generative, 0-shot — no exemplars, so the prompt has to carry the format

Two different things live here. The first line is the 0-shot counterpart of set A's `greekmmlu` (raw prompt with a
`\boxed{}` instruction, parses ≈100% for base and fine-tuned models alike; the exception is Meltemi-7B-base, which
degenerates into newlines). The rest are the open-ended tasks, run with the chat template for instruct models and the
raw prompt for bases.

| shots | tasks | prompt | `run_suite --groups` |
|---:|---|---|---|
| 0 | `greekmmlu_gen_boxed` | raw | `gen_mmlu_fs0` |
| 0 | `ilspgreekifeval` (+ English `ifeval`) | chat | `gen_fs0` |
| 0 | `ilspgreekcivicsqa` | chat | `gen_civics` |
| 0 | `ilspgreekflores_en_el`, `ilspgreekflores_el_en`, `ilspgreektruthfulqa_gen` | chat | `gen_trans` |
| 0 | `ilspgreekmmlupro` (`--limit 2000`; long chain-of-thought) | chat | `gen_mmlupro` |

`run_suite --raw_gen` re-runs the chat lines as raw completion into `<group>_raw/` for base models; never put raw and
chat numbers in the same column.

## Commands

Base template — set the model, the task and the shot count; keep `--log_samples` (it is the only record of what the
model wrote, and the diagnostic below reads it):

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
M='{"pretrained": "/path/to/model", "dtype": "bfloat16"}'          # add "trust_remote_code": true for K2
lm-eval --model hf --model_args "$M" --tasks <task> --num_fewshot <n> --batch_size 4 \
        --device cuda:0 --log_samples --seed 1234 --output_path runs/eval/<model>/<task>
```

`--num_fewshot` is global, so tasks with different shot counts go in separate calls. Batch 4 is safe for 5-shot MMLU on a
48 GB card (16 OOMs); 8 is fine for the letter-writing tasks.

### Multiple-choice (raw prompt, every model the same way)

```bash
lm-eval ... --tasks greekmmlu,belebele_ell_Grek,ilspgreekasep,ilspgreekmmlu,ilspgreekwinogrande --num_fewshot 5
lm-eval ... --tasks greekmmlu,ilspgreektruthfulqa_mc1,ilspgreektruthfulqa_mc2                    --num_fewshot 0
lm-eval ... --tasks ilspgreekmedicalmcqa                                                            --num_fewshot 15
lm-eval ... --tasks ilspgreekhellaswag                                                              --num_fewshot 10
lm-eval ... --tasks ilspgreekarc_easy,ilspgreekarc_challenge                                        --num_fewshot 25
lm-eval ... --tasks mmlu --num_fewshot 5 --limit 40                                                 # English guard, 40/subject
```

### Generative (chat template for instruct models)

```bash
CHAT='--apply_chat_template --gen_kwargs {"until":["<|im_end|>"]}'    # until = the model's turn-end token
lm-eval ... $CHAT --tasks ilspgreekifeval,ilspgreekcivicsqa,ilspgreekflores_en_el,ilspgreekflores_el_en,ilspgreektruthfulqa_gen --num_fewshot 0
lm-eval ... $CHAT --tasks ilspgreekmgsm --num_fewshot 8 --fewshot_as_multiturn true
lm-eval ... $CHAT --tasks ilspgreekmmlupro --num_fewshot 0 --limit 2000     # 12,032 items × long CoT: cap it
```

Reasoning-style templates open a think block by default — close it or every generative score is wrong: Qwen3.5 takes
`"enable_thinking": false` in `--model_args`; K2-Horizon takes `"chat_template_suffix": "</ifm|think>"` (this fork's
addition) and `--fewshot_as_multiturn false`. A base model that does not understand its template takes the raw prompt
instead (drop `$CHAT`); expect the format-related failures the diagnostic below reports.

### Letter-writing re-scoring of the MCQ tasks (raw prompt, same as MCQ)

```bash
lm-eval ... --tasks greekmmlu_gen,ilspgreekmmlu_gen,belebele_ell_Grek_gen --num_fewshot 5 --batch_size 8
lm-eval ... --tasks greekmmlu_gen_boxed                                    --num_fewshot 0 --batch_size 8
lm-eval ... --tasks mmlu_generative --num_fewshot 5 --limit 40
```

Compare each with the log-likelihood task at the **same shot count**: `greekmmlu_gen` ↔ `greekmmlu` 5-shot,
`greekmmlu_gen_boxed` ↔ `greekmmlu` 0-shot. Measured on 11 models: at 5-shot the two agree within 0.7 pt on every Greek
benchmark; at 0-shot fine-tuned models score 2–4 pt higher when they write the letter than under log-likelihood.

### Generation diagnostic (training repo)

```bash
python -m greekllm.eval.failure_profile --runs base=runs/eval/base sft=runs/eval/sft --tokenizer /path/to/model
```

Reads the `--log_samples` dumps; reports termination, empty replies, language (GlotLID), repetition, and IFEval's
format pass rate. It is a diagnostic, not a benchmark.

## Reading the numbers

- A generative `0.000` almost always means empty replies, not a wrong model: check `parsed` (letter tasks) or the
  diagnostic's `empty` row before believing it.
- Report MCQ numbers as raw few-shot log-likelihood when comparing with Meltemi / Llama-Krikri / the Open LLM
  Leaderboard; that is the protocol they use.
- `ilspgreekmedicalmcqa` evaluates on the `train` split — that is ILSP's own configuration and what their 48.0% means.
- Machine-translated benchmarks (ILSP MMLU, ARC, HellaSwag, MGSM, WinoGrande) carry translation noise and contamination
  risk; the native ones (GreekMMLU, medical, ASEP, civics) and the professionally translated ones (Belebele, FLORES)
  are the ones to weight.
