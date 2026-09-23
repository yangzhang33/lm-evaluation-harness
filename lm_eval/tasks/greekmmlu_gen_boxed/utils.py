"""Generative GreekMMLU with an explicit answer-format instruction, for 0-shot use.

Without exemplars a base model has nothing telling it to answer with a letter: at 0-shot Qwen3.5-4B-Base
produced a parseable letter for only 87% of items under the plain prompt, and lost 11 points against its own
log-likelihood score on the same items; with this instruction it parsed 100% and matched it (+0.6).
The wording is the one used for the earlier boxed-answer runs on this benchmark, so numbers are comparable
with those."""

from lm_eval.tasks.greekmmlu.utils import LABELS, subjects_gr

TO_GREEK = {**dict(zip("ABCD", "ΑΒΓΔ")), **dict(zip("abcd", "ΑΒΓΔ")), **dict(zip("αβγδ", "ΑΒΓΔ"))}


def doc_to_text(doc):
    letters = [LABELS[i][0] for i in range(len(doc["choices"]))]
    options = "\n".join(f"{LABELS[i]} {choice}" for i, choice in enumerate(doc["choices"]))
    forms = " ή ".join(f"\\boxed{{{letter}}}" for letter in letters)
    return (
        f"Αυτό είναι μια ερώτηση {subjects_gr.get(doc['subject'], doc['subject'])}. Επίλεξε τη σωστή απάντηση.\n\n"
        f"Ερώτηση: {doc['question']}\n{options}\n\n"
        "Απάντησε μόνο με το γράμμα της σωστής επιλογής μέσα σε πλαίσιο. "
        f"Χρησιμοποίησε ακριβώς μία από τις εξής μορφές: {forms}. Μην προσθέσεις άλλο κείμενο.\n\n"
        "Απάντηση:"
    )


def doc_to_target(doc):
    return LABELS[doc["answer"]][0]


def process_results(doc, results):
    pred = results[0]
    parsed = pred != "[invalid]"
    letter = TO_GREEK.get(pred, pred) if parsed else None
    return {"exact_match": float(parsed and letter == doc_to_target(doc)), "parsed": float(parsed)}
