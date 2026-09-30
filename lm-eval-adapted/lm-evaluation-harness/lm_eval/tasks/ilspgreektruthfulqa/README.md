# ILSP Greek TruthfulQA

Machine-translated Greek version of TruthfulQA, released by ILSP.

* **Dataset:** https://huggingface.co/datasets/ilsp/truthful_qa_greek
* **Reference implementation:** `truthfulqa_el` in `community_tasks/greek_evals.py` of
  https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreektruthfulqa_mc1`, `ilspgreektruthfulqa_mc2`, `ilspgreektruthfulqa_gen`
  (tag: `ilspgreektruthfulqa`)

Evaluated 0-shot on the dataset's `train` split (the only split it ships). ILSP reports
TruthfulQA MC2.

### Prompt and deviations from the reference

The 6 primer QA pairs are ILSP's, with the `Ερώτηση:` / `Απάντηση:` cue. Three mechanical
defects in the reference implementation are **not** reproduced:

1. Its primer is built from an indented Python triple-quoted string, so the rendered
   prompt carries stray newlines and 6-space indents on every answer line.
2. Its multiple-choice primer writes `Απάντηση` with a Latin `A` (U+0041); the generation
   primer uses the Greek `Α` (U+0391). Here it is Greek everywhere.
3. Its generation primer truncates the telescope answer mid-sentence
   (`...και να κάνουν τα>`). The complete sentence from the multiple-choice primer is
   used instead.

The query line also ends with `Απάντηση:` rather than the reference's colon-less
`Απάντηση`, so it matches the primer.

Because of these fixes — and because lighteval scores mc1/mc2 jointly from one combined
choice list while lm-eval keeps them as separate tasks — numbers here are comparable in
setup to ILSP's published results, but not expected to match them exactly.

### Changes to `utils.py`

It is the upstream `truthfulqa` one with two changes, both needed for Greek:

1. The fallback answer appended to `correct_answers` is `Δεν έχω κανένα σχόλιο.`, the
   phrasing used by the dataset and by the primer, instead of the English
   `I have no comment.`. Without this, a correctly refusing model is scored against an
   English reference it can never match.
2. `RougeScorer` is given a Unicode-aware tokenizer. `rouge_score`'s default replaces
   every character outside `[a-z0-9]` with a space, so Greek text tokenizes to an empty
   list and every ROUGE score comes out exactly 0 — even when the prediction is identical
   to the reference. BLEU is unaffected, since sacrebleu is called with `tokenize="intl"`.
   ROUGE numbers here are therefore not comparable to ROUGE numbers from the English
   `truthfulqa` task, which uses the default tokenizer.
