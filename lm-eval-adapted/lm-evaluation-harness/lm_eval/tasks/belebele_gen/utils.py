"""Generative Belebele (Greek): the upstream `belebele_ell_Grek` prompt unchanged, the model writes the letter."""

# The prompt labels options with Latin A-D; a Greek letter in the answer is mapped onto them by position.
TO_LATIN = {**dict(zip("ΑΒΓΔ", "ABCD")), **dict(zip("αβγδ", "ABCD")), **dict(zip("abcd", "ABCD"))}


def doc_to_target(doc):
    return "ABCD"[int(doc["correct_answer_num"]) - 1]


def process_results(doc, results):
    pred = results[0]
    parsed = pred != "[invalid]"
    letter = TO_LATIN.get(pred, pred) if parsed else None
    return {"exact_match": float(parsed and letter == doc_to_target(doc)), "parsed": float(parsed)}
