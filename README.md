# evaluation/

The Greek evaluation toolkit: the lm-eval fork with the Greek tasks, and the tools that run a campaign with it. It
holds no model list and no results. Those belong to the project that uses it: the project supplies a model registry
and a root folder, and the toolkit writes its outputs there.

## Docs

| doc | read it for |
|---|---|
| this page | what is here, install, the model registry, tool commands, paths |
| [`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md) | the one-page reference: every task, the groups the driver runs (shots, caps, prompt), the output format, thinking settings, plain lm-eval commands |
| [`GREEK_EVAL.md`](GREEK_EVAL.md) ([中文](GREEK_EVAL.zh.md)) | the background: task sources, protocol trade-offs, known pitfalls, what is not included and why |

## What is here

| path | what it is |
|---|---|
| `lm-eval-adapted/lm-evaluation-harness/` | the lm-eval fork with the Greek tasks; its own git repo (`yangzhang33/lm-evaluation-harness`, branch `el`). Each Greek task directory has a README on how it differs from its reference implementation |
| `greekeval/run_suite.py` | campaign driver: every model × every task group, resumable |
| `greekeval/models.py` | loads the project's model registry (see "Model registry") |
| `greekeval/collect.py`, `export_scores.py` | headline tables (`summary*.csv/md`) and the long table of every score (`all_scores.csv`) |
| `greekeval/status.py` | what is done and what is running, read off the filesystem and `/proc` |
| `greekeval/failure_profile.py` | generation diagnostics over the logged samples (empty, wrong language, loops, truncation) |
| `greekeval/judge/` | LLM judge: `prepare` (prompt sets), `generate` (collect answers), `judge` (score them) |
| `greekeval/verify_model_dir.py` | checks a local model directory is complete before GPU hours are spent on it |
| `greekeval/paths.py` | how every path is resolved (see "Paths") |
| `greekeval/langid.py`, `templates.py`, `log.py` | small helpers (language ID, chat-message preparation, logging) |
| `models.example.yaml` | example model registry |
| `ilsp-assets/` | ILSP's Greek MT-Bench judge prompts |

## Install

```bash
pip install -e evaluation/lm-eval-adapted/lm-evaluation-harness              # the fork; adds ~28 packages, touches no torch/transformers pin
pip install langdetect immutabledict                                          # ilspgreekifeval
python -c "import nltk; nltk.download('punkt_tab')"                           # Greek sentence splitting, ~11 MB
pip install -e evaluation --no-deps --no-build-isolation                      # this toolkit
```

`greekeval` declares no dependencies and uses what the environment already has (torch, transformers, pyyaml,
pyarrow, tokenizers, fasttext; `anthropic` or `openai` only for a real judge run). `ilspgreektruthfulqa_gen` needs
`sacrebleu` and `rouge_score`, which the fork's install brings. Datasets download from the Hub on first use;
afterwards run everything with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. Every model is a local path.

## Model registry

The models are not part of the toolkit. List them in a YAML file in your project, `configs/eval/models.yaml` under
the project root (or any file named by `$GREEKEVAL_MODELS`). Start from `models.example.yaml`, which documents the
fields: local path, turn-end token, how the think block is closed, batch policy, judge budget. The key of a model is
what you pass to `--models` and the name of its output folder; table columns follow the order of the file.

## Commands

```bash
python -m greekeval.run_suite --list --models <key>                     # the plan for one model
python -m greekeval.run_suite --gpu 0 --models <key> --groups mcq_fs5 gen_fs8b
python -m greekeval.run_suite --gpu 0 --models <key> --raw_gen          # chat groups as raw completion -> <group>_raw/
python -m greekeval.run_suite --gpu 0 --models <key> --chat_mcq --groups gen_mmlu_fs5 gen_mmlu_fs0   # letter groups with the chat template -> <group>_chat/
python -m greekeval.run_suite --gpu 0 --models <key> --think --out_root runs/eval_think    # think block open

python -m greekeval.collect --csv reports/eval_results/summary.csv --md reports/eval_results/summary.md
python -m greekeval.collect --protocol raw --csv reports/eval_results/summary_raw.csv --md reports/eval_results/summary_raw.md
python -m greekeval.export_scores                                       # -> reports/eval_results/all_scores.csv
python -m greekeval.status                                              # --watch 300 keeps runs/eval/STATUS.md
python -m greekeval.failure_profile --runs base=runs/eval/<key> sft=runs/eval/<key> --tokenizer <model dir>
python -m greekeval.verify_model_dir <model dir>

python -m greekeval.judge.prepare                                       # -> data/judge/<set>.jsonl
python -m greekeval.judge.generate --gpu 0 --models <key> [<key> ...]
python -m greekeval.judge.judge --provider dry --models <key>           # renders the judge prompts, calls nothing
```

Which groups to run, how many shots, which token caps and which prompt: [`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md).

## Paths

A relative path is always taken from the **project root**, never from the current directory. That covers the model
paths in the registry, `--out_root`, `--root`, `--runs`, and output files such as `--csv`. An absolute path is used
as is.

The project root is the folder above `evaluation/`. Set `GREEKEVAL_PROJECT=/some/dir` to use another one: the
registry, `models/`, `runs/`, `data/` and `reports/` are then looked up there. The lm-eval fork is always the one
next to this package.

Outputs: `runs/eval/`, `runs/eval_think/`, `runs/judge/`, `data/judge/`, `reports/eval_results/`, all under the
project root.

## Output format

The folder layout (`runs/eval/<model>/<group><suffix>/<sanitized model path>/`) and the rules every run must follow
are defined in one place: [`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md#output-format), section "Output format".
