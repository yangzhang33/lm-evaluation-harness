import re


# The option labels the dataset is written against: Latin letters, as in the
# original MMLU-Pro, even though the questions and options are Greek.
CHOICE_LABELS = "ABCDEFGHIJKLMNOP"

# ILSP's instruction, verbatim apart from the whitespace artifacts of the Python
# triple-quoted string it lives in. Two of their typos are kept so the prompt matches
# the one they published with: "μαζί με της απαντήσεις" (should be "τις") and
# "αντιστοιχτεί" (should be "αντιστοιχεί"). The placeholder "(Χ)" is a Greek capital
# chi; the letter the model has to produce is a Latin one from the option list.
INSTRUCTION = (
    "Οι ακόλουθες ερωτήσεις πολλαπλής επιλογής παρουσιάζονται μαζί με της απαντήσεις τους.\n"
    'Σκέψου βήμα προς βήμα και τέλειωσε την απάντηση σου με "η απάντηση είναι (Χ)" '
    "όπου Χ είναι το γράμμα που αντιστοιχτεί στην σωστή επιλογή.\n"
)


def doc_to_text(doc):
    prompt = INSTRUCTION + "Ερώτηση:\n" + doc["question"] + "\n" + "Επιλογές:\n"
    for label, option in zip(CHOICE_LABELS, doc["options"]):
        prompt += f"{label}. {option}\n"
    return prompt + "Απάντηση: "


def extract_answer(text):
    """ILSP's three-stage answer extraction."""
    match = re.search(r"απάντηση είναι \(?([A-P])\)?", text)
    if match:
        return match.group(1)
    match = re.search(r".*[αΑ]πάντηση:\s*\(?([A-P])\)?", text)
    if match:
        return match.group(1)
    # Last standalone option letter anywhere in the response.
    match = re.search(r"\b[A-P]\b(?!.*\b[A-P]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def process_results(doc, results):
    prediction = extract_answer(results[0])
    return {"exact_match": float(prediction == doc["answer"])}
