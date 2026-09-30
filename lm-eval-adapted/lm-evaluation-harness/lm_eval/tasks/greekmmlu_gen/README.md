# GreekMMLU, generative

The same 16,632 native GreekMMLU questions as `greekmmlu`, scored by having the model **write** the option letter
instead of ranking the log-likelihood of ` Α` / ` Β` / ` Γ` / ` Δ`. Log-likelihood scoring has two known blind spots —
a preference for a letter or position regardless of content, and a shift in the calibration of single-letter
continuations after chat-style fine-tuning — and this task exists to check whether a score moved because of those
or because knowledge changed.

### Groups and tasks

Mirrors `greekmmlu` exactly: 45 subject tasks `greekmmlu_gen_<subject>`, the four category groups
`greekmmlu_gen_stem` / `_humanities` / `_social_sciences` / `_other`, and the top-level group `greekmmlu_gen`.
Configs are derived from the sibling directory by `_generate_configs.py`; re-run it after any change there.

### Protocol

- **Prompt: byte-identical to `greekmmlu`.** Same `utils.doc_to_text`, same Greek option labels, same trailing
  ` Απάντηση:`; few-shot exemplars from the same `dev` split in the same order, rendered as `Απάντηση: Δ`.
  Verified against sample dumps of `greekmmlu`.
- **Generation:** greedy, 8 new tokens, no stop string (the extractor only looks at the start).
- **Extraction (`letter` filter):** the first standalone option letter, optionally in `(`/`[` and followed by
  punctuation. A letter that begins a word — `Απάντηση`, `Δεν` — is not an answer. Greek `Α-Δ` and Latin `A-D`
  (either case) are accepted; Latin is mapped onto Greek by position, since a model writing `C` for the third
  option knows the answer and merely switched script.
- **Metrics:** `exact_match` (an unparsed answer is wrong) and `parsed` (share of items that yielded a letter), so
  a drop in the first can be read as knowledge or as format. Result keys: `exact_match,letter`, `parsed,letter`.

### Use it 5-shot; use `greekmmlu_gen_boxed` for 0-shot

With five exemplars the format is unambiguous and the score tracks log-likelihood on the very same items.

At 0-shot nothing tells the model to answer with a letter. A chat-tuned model usually still does; a base model
tends to write `Η σωστή απάντηση είναι …`, restate `Απάντηση:`, or give the answer text, and is scored wrong for
it. That is a format effect, not knowledge, and it is not evenly distributed across models — so for 0-shot use
`greekmmlu_gen_boxed`, which adds an explicit answer-format instruction that base and fine-tuned models follow
alike.

Run it under the multiple-choice protocol: raw prompt, no chat template, `--num_fewshot 5`. Compare with the
5-shot `greekmmlu` run, not the 0-shot one.
