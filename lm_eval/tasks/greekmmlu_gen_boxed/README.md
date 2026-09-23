# GreekMMLU, generative, boxed answer (0-shot)

`greekmmlu_gen` with one addition: an explicit instruction to answer with the option letter inside `\boxed{}`.
Built for **0-shot**, where the plain prompt gives a base model nothing to infer the answer format from.

### Groups and tasks

Same shape as `greekmmlu` and `greekmmlu_gen`: 45 subject tasks `greekmmlu_gen_boxed_<subject>`, four category
groups, top-level group `greekmmlu_gen_boxed`. Derived by `_generate_configs.py` from the sibling `greekmmlu/`.

### Protocol

Prompt (subject line and options as in `greekmmlu`, then the instruction; `\boxed{…}` forms listed for however
many options the item has):

```
Αυτό είναι μια ερώτηση Λογιστικής. Επίλεξε τη σωστή απάντηση.

Ερώτηση: <question>
Α. <option>
…
Δ. <option>

Απάντησε μόνο με το γράμμα της σωστής επιλογής μέσα σε πλαίσιο. Χρησιμοποίησε ακριβώς μία από τις εξής
μορφές: \boxed{Α} ή \boxed{Β} ή \boxed{Γ} ή \boxed{Δ}. Μην προσθέσεις άλλο κείμενο.

Απάντηση:
```

The wording is the one used for the earlier boxed-answer runs on this benchmark, kept verbatim so that numbers
stay comparable with those. It is **not** byte-identical to the `greekmmlu` prompt (`.` instead of `!` after
`Επίλεξε τη σωστή απάντηση`, no leading space before the final `Απάντηση:`, and the added paragraph), so treat a
comparison against the 0-shot log-likelihood score as approximate; the clean comparison is model against model
under this protocol.

- **Generation:** greedy, up to 64 new tokens, stopped at any common end-of-turn token
  (`<|im_end|>`, `<|endoftext|>`, `<|eot_id|>`, `<|ifm|endoftext|>`, `</s>`).
- **Extraction (`boxed` filter):** the letter inside the first `\boxed{…}`. Greek and Latin letters accepted,
  Latin mapped onto Greek by position. Anything without a box is unparsed and counts as wrong.
- **Metrics:** `exact_match` and `parsed`; result keys `exact_match,boxed`, `parsed,boxed`.

### Why the instruction is worth the prompt change

0-shot, 180 items per model, against the log-likelihood score on the same items:

| model | plain prompt (`greekmmlu_gen`) | boxed instruction |
|---|---|---|
| Qwen3.5-4B-Base | parse 86.7%, −11.1 pt | parse 100%, +0.6 pt |
| Qwen3.5-4B SFT (e3) | parse 100%, ±0.0 pt | parse 100%, +4.4 pt |
| K2-Horizon-3.7B base | parse 96.1%, −1.7 pt | parse 100%, +0.6 pt |

Base models follow the instruction as reliably as the fine-tuned one, so it removes the format penalty without
tilting the comparison. Few-shot is possible (`fewshot_split: dev` is configured) but every exemplar then repeats
the instruction paragraph; for few-shot use `greekmmlu_gen`.
