# ILSP Greek ASEP MCQA

Natively sourced Greek multiple-choice questions from ASEP civil-service entrance exams.

* **Dataset:** https://huggingface.co/datasets/ilsp/mcqa_greek_asep (2346 rows, split `default`)
* **Reference implementation:** `mcqa_asep` in https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekasep` — scored as the log-likelihood of the full option text, same prompt
  shape as `ilspgreekmedicalmcqa`. Option labels in the data are lowercase `α./β./γ./δ.`.

`answer` is already a 0-based integer index, used directly as the target.
