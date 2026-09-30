# Greek evaluation toolkit

The lm-eval fork with the Greek tasks, and the tools that run a campaign with it, in one repository. It holds no
model list and no results: you list your models in a local, git-ignored file, and the outputs go to the project
folder you name there.

## Docs

| doc | read it for |
|---|---|
| this page | what is here, install, the model registry, tool commands, paths |
| [`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md) | the rules: every task and what it measures, the output format, the thinking settings, how a plan line maps to lm-eval flags. Which runs exist is in `plan.yaml`, not there |
| [`GREEK_EVAL.md`](GREEK_EVAL.md) ([中文](GREEK_EVAL.zh.md)) | the background: task sources, protocol trade-offs, known pitfalls, what is not included and why |

## What is here

| path | what it is |
|---|---|
| `lm-eval-adapted/lm-evaluation-harness/` | lm-evaluation-harness with the Greek tasks added (`lm_eval/tasks/ilspgreek*`, `greekmmlu*`, `belebele_gen`) and a few fixes for reasoning-style chat templates; a plain subfolder of this repository. Each Greek task directory has a README on how it differs from its reference implementation |
| `greekeval/run_suite.py` | campaign driver: every model × every line of the plan, resumable |
| `config/plan.yaml` | the plan: every run the toolkit knows, one line each; comment out what you do not want (see "Plan") |
| `config/models.yaml` | your model registry; git-ignored, start from `config/models.example.yaml` (see "Model registry") |
| `greekeval/models.py` | loads the project's model registry (see "Model registry") |
| `greekeval/collect.py`, `export_scores.py` | headline tables (`summary*.csv/md`) and the long table of every score (`all_scores.csv`) |
| `greekeval/status.py` | what is done and what is running, read off the filesystem and `/proc` |
| `greekeval/failure_profile.py` | generation diagnostics over the logged samples (empty, wrong language, loops, truncation) |
| `greekeval/judge/` | LLM judge: `prepare` (prompt sets), `generate` (collect answers), `judge` (score them) |
| `greekeval/verify_model_dir.py` | checks a local model directory is complete before GPU hours are spent on it |
| `greekeval/paths.py` | how every path is resolved (see "Paths") |
| `greekeval/langid.py`, `templates.py`, `log.py` | small helpers (language ID, chat-message preparation, logging) |
| `ilsp-assets/` | ILSP's Greek MT-Bench judge prompts |

## Install

```bash
pip install -e lm-eval-adapted/lm-evaluation-harness                         # the fork; adds ~28 packages, touches no torch/transformers pin
pip install langdetect immutabledict                                          # ilspgreekifeval
python -c "import nltk; nltk.download('punkt_tab')"                           # Greek sentence splitting, ~11 MB
pip install -e . --no-deps --no-build-isolation                               # this toolkit, from the repository root
```

`greekeval` declares no dependencies and uses what the environment already has (torch, transformers, pyyaml,
pyarrow, tokenizers, fasttext; `anthropic` or `openai` only for a real judge run). `ilspgreektruthfulqa_gen` needs
`sacrebleu` and `rouge_score`, which the fork's install brings. Datasets download from the Hub on first use;
afterwards run everything with `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. Every model is a local path.

## Model registry

All configuration lives in `config/`, inside this folder. The models are listed in `config/models.yaml`, which is
git-ignored: it holds local paths, so it never leaves your machine (or use any file named by `$GREEKEVAL_MODELS`).
Start from `config/models.example.yaml`, which documents the
fields: local path, turn-end token, how the think block is closed, batch policy, judge budget. The same file names
your `project_root` (see "Paths"). The key of a model is
what you pass to `--models` and the name of its output folder; table columns follow the order of the file.

On another machine keep the keys and change only the paths: the key is what makes results from two machines land in
the same folder and the same table column. A machine-local copy of the registry is selected with `$GREEKEVAL_MODELS`.

## Plan

What gets run is not decided on the command line but in the plan file: `config/plan.yaml` lists every run the toolkit
knows (log-likelihood MCQ, letter-writing MCQ, open-ended, reasoning; raw prompt or chat template; think block closed
or open), one line each, with its tasks, shots, cap and output folder. Comment out the lines you do not want; the
driver runs the rest for the models you name and skips a run whose folder already has results. The file's header
explains every field. `--groups <name> ...` narrows one call to the named lines; `--plan <file>` reads another file.

## Commands

```bash
python -m greekeval.run_suite --list --models <key>                     # every line of the plan: done or todo
python -m greekeval.run_suite --gpu 0 --models <key>                    # run them all; skips what is done
python -m greekeval.run_suite --gpu 0 --models <key> --groups mcq_fs5 gen_fs8b   # only these lines
python -m greekeval.run_suite --dp 2 --models <key> --groups gen_mmlu_fs0        # one run over both GPUs
python -m greekeval.run_suite --gpu 0 --models <key> --limit 1 --runs runs/smoke # smoke test: 1 item per task, same layout under runs/smoke/

python -m greekeval.collect --csv runs/eval/summary.csv --md runs/eval/summary.md
python -m greekeval.collect --protocol raw --csv runs/eval/summary_raw.csv --md runs/eval/summary_raw.md
python -m greekeval.export_scores                                       # -> reports/eval_results/all_scores.csv
python -m greekeval.status                                              # --watch 300 keeps runs/eval/STATUS.md; --runs runs/smoke for a smoke folder
python -m greekeval.failure_profile --runs base=runs/eval/<key> sft=runs/eval/<key> --tokenizer <model dir>
python -m greekeval.verify_model_dir <model dir>

python -m greekeval.judge.prepare                                       # -> data/judge/<set>.jsonl
python -m greekeval.judge.generate --gpu 0 --models <key> [<key> ...]
python -m greekeval.judge.judge --provider dry --models <key>           # renders the judge prompts, calls nothing
```

Which groups to run, how many shots, which token caps and which prompt: [`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md).

## Paths

A relative path is always taken from the **project root**, never from the current directory. That covers the model
paths in the registry, `--runs`, `--root`, and output files such as `--csv`. An absolute path is used as is.

The project root is where `models/`, `runs/`, `data/` and `reports/` live. Set it as `project_root` in
`config/models.yaml`; `$GREEKEVAL_PROJECT` overrides it. Without either, it is this folder itself, and the outputs
(`runs/`, `data/`) are git-ignored here. The lm-eval fork is always the one in this repository.

Outputs: `runs/eval/`, `runs/eval_think/`, `runs/judge/`, `data/judge/`, `reports/eval_results/`, all under the
project root.

## Output format

The folder layout (`runs/eval/<model>/<group><suffix>/<sanitized model path>/`) and the rules every run must follow
are defined in one place: [`GREEK_BENCHMARKS.md`](GREEK_BENCHMARKS.md#output-format), section "Output format".
