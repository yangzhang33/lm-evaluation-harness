# ILSP assets kept out of the harness

Files lifted from ILSP's lighteval fork (https://github.com/LeonVouk/lighteval,
Apache-2.0) that are useful but do not belong to an lm-eval task.

- `judge_prompt_el_templates.py` — the Greek MT-Bench judge prompts (with and without a
  reference answer, single-turn and multi-turn). Needed if the Greek MT-Bench is ever
  built as a standalone judge script; MT-Bench is multi-turn plus a judge, which does not
  fit lm-eval.

Everything else from that fork is either already ported into
`lm-eval-adapted/lm-evaluation-harness/lm_eval/tasks/ilspgreek*` or was a prompt function
short enough to reimplement. The clone itself has been deleted.
