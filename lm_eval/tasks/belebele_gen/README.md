# Belebele (Greek), generative

`belebele_ell_Grek` — professionally translated reading comprehension, 900 items — scored by having the model
**write** the answer letter instead of ranking the log-likelihood of ` A` / ` B` / ` C` / ` D`.

### Tasks

* `belebele_ell_Grek_gen`

### Protocol

- **Prompt: byte-identical to upstream `belebele_ell_Grek`** (`P:` passage, `Q:` question, `A:`–`D:` options,
  `Answer:`; five exemplars from other test passages, first-n, as upstream). Verified against the campaign's sample
  dumps.
- **Generation:** greedy, 8 new tokens, no stop string.
- **Extraction (`letter` filter):** the first standalone option letter. The labels here are Latin, so a Greek
  `Α-Δ` in the answer is mapped onto `A-D` by position.
- **Metrics:** `exact_match` (unparsed = wrong) and `parsed`. Result keys `exact_match,letter`, `parsed,letter`.

Run it as the multiple-choice task is run: raw prompt, no chat template, `--num_fewshot 5`. Smoke on
Qwen3.5-4B-Base, first 40 items: generative 92.5%, log-likelihood on the same items 92.5%, parsed 100%.
