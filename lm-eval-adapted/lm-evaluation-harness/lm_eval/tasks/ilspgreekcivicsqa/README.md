# ILSP Greek Civics QA

Free-form Greek question answering over civics and politics, with human reference answers
taken from Greek secondary-school textbooks.

* **Dataset:** https://huggingface.co/datasets/ilsp/greek_civics_qa (407 rows, split `default`)
* **Reference implementation:** `greek_civics_qa` in https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekcivicsqa` — 0-shot generative, corpus BLEU against the reference answer.
  Generation is capped at 100 tokens and stops at the first newline, matching the
  reference implementation.

### Caveat

BLEU against a single long textbook answer is a weak signal: the references run to several
hundred words while generation is capped at 100 tokens, so scores are low and compress the
differences between models. The dataset also ships two ChatGPT answers per question
(`chatGPT_answer_1`, `chatGPT_answer_2`) with human comments on them, which makes it well
suited to an LLM-judge setup with anchors — that is worth building separately rather than
reading too much into the BLEU number here.
