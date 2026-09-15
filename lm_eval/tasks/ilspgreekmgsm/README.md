# ILSP Greek MGSM

Greek MGSM — grade-school math word problems with chain-of-thought answers.

* **Dataset:** https://huggingface.co/datasets/ilsp/mgsm_greek (`test` 250, `train` 8)
* **Reference implementation:** `mgsm:el` in https://github.com/LeonVouk/lighteval

### Groups and Tasks

* `ilspgreekmgsm` — generative, native-CoT. Two filters, as upstream `mgsm_native_cot`:
  `strict-match` (the answer sentence `Η απάντηση είναι <number>`) and `flexible-extract`
  (the last number in the response, which is what the reference implementation scores).

The 8 `train` rows are the standard MGSM exemplars; run with `--num_fewshot 8`.

### A data quirk worth knowing

The `test` rows carry `answer` as an **empty string**, not null, and their `question`
lacks the `Ερώτηση: ` prefix that the `train` exemplars have. The conditional in
`doc_to_text`/`doc_to_target` therefore tests truthiness (`{% if answer %}`), not
`is not none` — with `is not none` every test row takes the wrong branch, which empties
the target and drops the prefix.

### Running it

Instruct models need `--apply_chat_template --fewshot_as_multiturn`. Without it,
Meltemi-7B-Instruct-v1.5 returned empty generations for 36 of 40 sampled rows and scored
0.0; with it, flexible-extract 0.25 / strict-match 0.20.
