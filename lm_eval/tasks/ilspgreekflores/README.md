# ILSP Greek FLORES-200 (en ↔ el)

Translation between English and Greek on the professionally translated FLORES-200
sentences.

* **Dataset:** https://huggingface.co/datasets/ilsp/flores200_en-el (`test` and `validation`)
* **Reference implementation:** `flores200:en->el` / `flores200:el->en` in
  https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekflores_en_el`, `ilspgreekflores_el_en` (tag: `ilspgreekflores`)

Few-shot examples come sequentially from `validation`, generation is capped at 100 tokens
and stops at the first newline — the reference implementation's settings.

### Metrics

Corpus BLEU, as ILSP reports, plus chrF. chrF is the more informative of the two for
Greek: it scores character n-grams, so it does not punish a correct translation for
picking a different inflected form. Read BLEU when comparing against ILSP's numbers and
chrF when comparing models with each other.

The prompts keep ILSP's spelling as published, including `απο` (for `από`) and the
unaccented `Αγγλικα:` in the en→el direction.

### Contamination

WMT-23 and later test sets must stay out of training data if these numbers are to mean
anything.
