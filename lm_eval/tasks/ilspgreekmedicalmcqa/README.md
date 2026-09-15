# ILSP Greek Medical MCQA

Natively sourced Greek medical multiple-choice questions from DOATAP medical licensing
exams — not a translation, so contamination risk is low.

* **Dataset:** https://huggingface.co/datasets/ilsp/medical_mcqa_greek
* **Reference implementation:** `medicalmcqa` in https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekmedicalmcqa` — 5 options, scored as the log-likelihood of the full option
  text. ILSP reports it 15-shot.

### Splits

Evaluated on `train` (1602), few-shot examples drawn sequentially from `validation` (432),
which is what the reference implementation does — despite the split names. Swap
`test_split` and `fewshot_split` if you would rather evaluate on the smaller held-out
half; the numbers will not be comparable with ILSP's published 48.0% if you do.

### Prompt

```
Ερώτηση: <question>

Επιλογές:
Α. <option 1>
...
Ε. <option 5>

Απάντηση:
```

The options already carry their `Α./Β./Γ./Δ./Ε.` labels in the data. The gold is the index
where `multiple_choice_scores` is 1.0.
