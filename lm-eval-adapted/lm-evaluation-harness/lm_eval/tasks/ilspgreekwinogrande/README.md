# ILSP Greek WinoGrande (filtered)

Machine-translated Greek version of WinoGrande, released by ILSP as pairs of complete sentences.

* **Dataset:** https://huggingface.co/datasets/ilsp/winogrande_greek
* **Reference implementation:** none. ILSP's own lighteval suite has no winogrande task; upstream lm-eval's
  `winogrande` scheme (shared context, score the differing continuation) is followed on the Greek sentences.

### Groups and Tasks

* `ilspgreekwinogrande`

Evaluated on `validation` (1,267 rows, 1,021 after filtering), few-shot sampled from the filtered `train`.

### Deviations

`sentence`, `option1` and `option2` stay **English** in this dataset; only `multiple_choice_targets` is Greek, and its
two candidates were translated independently. Two artifact classes are removed in `process_docs` (rates over the full
validation split): pairs whose common word-prefix is under half the shorter sentence, i.e. the whole sentence was
retranslated differently (18.9%), and pairs where a Latin-script name survives in exactly one branch (0.5%). What is
left contrasts only at the substituted entity and what follows it, which is the WinoGrande task. Scores are not
comparable to any number computed on the unfiltered pairs.
