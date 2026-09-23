"""Generative GreekMMLU: the same prompt as `greekmmlu`, but the model writes the option letter instead of the
harness ranking the log-likelihood of " Α" / " Β" / " Γ" / " Δ"."""

# The prompt is byte-identical to the log-likelihood task by construction: same function, same subject names,
# same "Α. / Β. / Γ. / Δ." option lines, same trailing " Απάντηση:".
from lm_eval.tasks.greekmmlu.utils import LABELS, doc_to_text  # noqa: F401

# Models sometimes answer in the Latin alphabet ("C" for the third option) or in lowercase; both are the same
# answer, so they are mapped onto the Greek label by position before comparing.
TO_GREEK = {**dict(zip("ABCD", "ΑΒΓΔ")), **dict(zip("abcd", "ΑΒΓΔ")), **dict(zip("αβγδ", "ΑΒΓΔ"))}


def doc_to_target(doc):
    """The gold option letter. `greekmmlu` returns the option *index*, which is what a multiple-choice task
    needs; a generative task would render that as "Απάντηση: 0" in the few-shot exemplars."""
    return LABELS[doc["answer"]][0]


def process_results(doc, results):
    """`results[0]` is what the yaml filter extracted: a single letter, or "[invalid]" when the generation did
    not start with one. An unparsed answer is wrong; `parsed` is reported alongside so that a drop in
    `exact_match` can be read as knowledge or as format."""
    pred = results[0]
    parsed = pred != "[invalid]"
    letter = TO_GREEK.get(pred, pred) if parsed else None
    return {"exact_match": float(parsed and letter == doc_to_target(doc)), "parsed": float(parsed)}
