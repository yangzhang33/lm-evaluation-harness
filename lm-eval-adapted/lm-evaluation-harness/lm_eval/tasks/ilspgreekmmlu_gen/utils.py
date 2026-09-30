"""Generative ILSP Greek MMLU: the `ilspgreekmmlu` prompt unchanged, the model writes the option letter."""

TO_GREEK = {**dict(zip("ABCD", "ΑΒΓΔ")), **dict(zip("abcd", "ΑΒΓΔ")), **dict(zip("αβγδ", "ΑΒΓΔ"))}


def doc_to_target(doc):
    return "ΑΒΓΔ"[doc["answer"]]


def process_results(doc, results):
    """`results[0]` is the letter the yaml filter extracted, or "[invalid]". Latin letters are mapped onto the
    Greek labels by position; an unparsed answer is wrong and `parsed` is reported alongside."""
    pred = results[0]
    parsed = pred != "[invalid]"
    letter = TO_GREEK.get(pred, pred) if parsed else None
    return {"exact_match": float(parsed and letter == doc_to_target(doc)), "parsed": float(parsed)}
