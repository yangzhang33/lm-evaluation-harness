# ILSP Greek ARC

Machine-translated Greek version of AI2 ARC, released by ILSP.

* **Dataset:** https://huggingface.co/datasets/ilsp/arc_greek
* **Reference implementation:** `arc_el` in `community_tasks/greek_evals.py` of
  https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekarc_easy`, `ilspgreekarc_challenge` (tag: `ilspgreekarc`)

Evaluated on `test`, few-shot sampled from `train`, matching the reference
implementation. ILSP reports ARC-Challenge 25-shot.

Prompt and metrics are the upstream `arc` ones with the question/answer cue translated
(`Ερώτηση:` / `Απάντηση:`), which is what the reference implementation uses.
