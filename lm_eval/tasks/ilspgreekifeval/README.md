# ILSP Greek IFEval

Greek port of IFEval — prompts carrying verifiable constraints ("answer in more than 400
words", "no commas", "wrap the answer in quotes"), checked by heuristics rather than a
judge.

* **Dataset:** https://huggingface.co/datasets/ilsp/ifeval_greek (541 rows, `train`)
* **Reference implementation:** `ifeval_el` in https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekifeval` — 0-shot, generative. Reports prompt-level and instruction-level
  accuracy, strict and loose.

### Where the code comes from

`instructions.py`, `instructions_registry.py` and `instructions_util.py` are ILSP's Greek
adaptation of the Google IFEval instruction library (Apache-2.0), taken from their
lighteval fork (`ifeval_el_instructions*.py`) — not the English files that ship with
lm-eval's `ifeval`. This matters:

- The dataset's `instruction_id_list` is written against ILSP's registry. Two of its ids,
  `change_case:capital` and `change_case:lowercase`, do not exist in the English registry,
  so the English module raises `KeyError` on them.
- Greek-sensitive checks are reimplemented there: casing (final sigma `ς`/`σ`, accents),
  letter frequency over the Greek alphabet, and sentence counting with the Greek Punkt
  model.

Two changes were made while porting:

1. Imports repointed from `lighteval.tasks.extended.ifeval` to this package.
2. `_get_sentence_tokenizer()` used `nltk.data.load("nltk:tokenizers/punkt/greek.pickle")`.
   nltk >= 3.9 refuses to load pickle-based punkt models (nltk#3266), so it now uses
   `nltk.tokenize.punkt.PunktTokenizer("greek")`, which loads the same Greek model from
   the `punkt_tab` data package. Requires `nltk.download("punkt_tab")`.

3. `keywords:letter_frequency` was still the English implementation and was broken for
   this dataset. `build_description` accepted a letter only if `97 <= ord(letter) <= 122`
   and otherwise **silently replaced it with a random ASCII letter** — and every letter
   the dataset asks about fails that test, both the Greek letters (`τ`, `ο`, …) and the
   punctuation (`#`, `!`). So all 33 occurrences were checking a random Latin character
   against a Greek response, non-deterministically. It now accepts any single character,
   and `check_following` folds Greek diacritics and final sigma (`instructions_util.fold_greek`)
   so that `ά` counts as `α` and `ς` as `σ`.

`utils.py` is lm-eval's own `ifeval/utils.py` with its import pointed at the Greek
registry.

### Dependencies

`langdetect`, `nltk` (>= 3.9) with the `punkt_tab` data package.

### Running it

Instruct models need `--apply_chat_template`. Without it they see a bare prompt and many
of them emit EOS immediately: on Meltemi-7B-Instruct-v1.5, 8 of 15 sampled generations
came back empty and instruction-level accuracy was 0.087, against 0.565 with the template.
Base models take the raw prompt, as usual.

### Caveat for reports

The SFT mix contains ~42K rows of `openeurollm/EU-Instruct-Synthetic`, the same
constrained-instruction genre. Scores on this benchmark are optimistic; say so.
