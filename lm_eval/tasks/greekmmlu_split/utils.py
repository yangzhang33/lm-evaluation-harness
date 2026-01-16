"""
Greek MMLU utility functions for formatting questions and choices.
"""
import re

# Regex to find answer letters (Greek or English)
ANSWER_RE = re.compile(r"[ΑΒΓΔABCD]")

PROMPT = (
    "Αυτό είναι μια ερώτηση {}.\n\n"
    "Ερώτηση: {}\n"
    "{}\n\n"
    "Απάντησε ΜΟΝΟ με το γράμμα της σωστής επιλογής "
    "(π.χ. Α, Β, Γ ή Δ).\n"
    "Απάντηση:"
)


# subjects_gr
subjects_gr = {
    "Economics": "Οικονομικών",
    "Education": "Παιδαγωγικής",
    "Medicine": "Ιατρικής",
    "Electrical Engineering": "Ηλεκτρολόγων Μηχανικών",
    "Greek Mythology": "Ελληνικής Μυθολογίας",
    "Computer Networks & Security": "Δικτύων Υπολογιστών και Ασφάλειας",
    "Law": "Νομικής",
    "Physics": "Φυσικής",
    "Government and Politics": "Διακυβέρνησης και Πολιτικής",
    "Art": "Τέχνης",
    "Greek Literature": "Νεοελληνικής Λογοτεχνίας",
    "World History": "Παγκόσμιας Ιστορίας",
    "General Knowledge": "Γενικών Γνώσεων",
    "World Religions": "Παγκόσμιων Θρησκειών",
    "Mathematics": "Μαθηματικών",
    "Clinical Knowledge": "Κλινικών Γνώσεων",
    "Driving Rules": "Κανόνων Οδικής Κυκλοφορίας",
    "Biology": "Βιολογίας",
    "Civil Engineering": "Πολιτικών Μηχανικών",
    "Computer Science": "Επιστήμης Υπολογιστών",
    "Geography": "Γεωγραφίας",
    "Chemistry": "Χημείας",
    "Prehistory": "Προϊστορίας",
    "Agriculture": "Γεωργίας",
    "Modern Greek Language": "Νεοελληνικής Γλώσσας",
    "Accounting": "Λογιστικής",
    "Greek History": "Ελληνικής Ιστορίας",
    "Management": "Διοίκησης Επιχειρήσεων",
    "Greek Traditions": "Ελληνικών Παραδόσεων",
}



# Greek choice labels
LABELS = ["Α.", "Β.", "Γ.", "Δ."]


def doc_to_text(doc):
    """
    Format the question with choices for Greek MMLU.
    
    Args:
        doc: Dictionary with 'question' and 'choices' fields
        
    Returns:
        Formatted question string
    """
    question = doc["question"]
    choices = doc["choices"]
    subject = doc["subject"]
    
    # Convert English subject to Greek
    subject_gr = subjects_gr.get(subject, subject)
    
    # Format choices with Greek labels
    formatted_choices = []
    for i, choice in enumerate(choices):
        formatted_choices.append(f"{LABELS[i]} {choice}")
    
    choices_text = "\n".join(formatted_choices)
    
    return PROMPT.format(subject_gr, question, choices_text)


def doc_to_choice(doc):
    """
    Extract choice labels based on number of choices.
    
    Args:
        doc: Dictionary with 'choices' field
        
    Returns:
        List of choice labels (e.g., ['Α', 'Β', 'Γ', 'Δ'])
    """
    num_choices = len(doc["choices"])
    return [LABELS[i][0] for i in range(num_choices)]


def process_results(doc, results):
    """
    Process results for generate_until format (API compatibility).
    Handles variable number of choices (2, 3, or 4).
    
    Args:
        doc: Dictionary with document data including 'answer' and 'choices' fields
        results: List containing the model's generated response
        
    Returns:
        Dictionary with accuracy metric
    """
    pred = results[0].strip().upper()

    num_choices = len(doc["choices"])
    valid_labels = [LABELS[i][0] for i in range(num_choices)]

    pred_letter = ""

    m = ANSWER_RE.search(pred)
    if m:
        c = m.group(0)

        if c in ["A", "B", "C", "D"]:
            mapping = {"A": "Α", "B": "Β", "C": "Γ", "D": "Δ"}
            c = mapping[c]

        if c in valid_labels:
            pred_letter = c

    gold = valid_labels[doc["answer"]]

    return {"acc": float(pred_letter == gold)}


def doc_to_target(doc):
    """
    Convert answer index to Greek letter string for fewshot examples.
    
    Args:
        doc: Dictionary with 'answer' field (integer index)
        
    Returns:
        Greek letter string (Α, Β, Γ, or Δ)
    """
    num_choices = len(doc["choices"])
    valid_labels = [LABELS[i][0] for i in range(num_choices)]
    return valid_labels[doc["answer"]]

