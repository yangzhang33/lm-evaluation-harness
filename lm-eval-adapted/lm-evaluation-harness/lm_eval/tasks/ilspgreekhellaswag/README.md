# ILSP Greek HellaSwag

Machine-translated Greek version of HellaSwag, released by ILSP.

* **Dataset:** https://huggingface.co/datasets/ilsp/hellaswag_greek
* **Reference implementation:** `hellaswag_el` in `community_tasks/greek_evals.py` of
  https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekhellaswag`

Evaluated on `validation` (the `test` split ships with empty `label` values), few-shot
sampled from `train`, matching the reference implementation. ILSP reports it 10-shot.

Context construction and `[title]`/`[header]` cleanup are identical to the upstream
`hellaswag` task; the translated Greek text keeps those English markers, so the same
preprocessing applies.
