
"""
Greek MMLU utility functions for formatting questions and choices.
Adapted from lm_eval/tasks/mmlu/flan_cot_zeroshot/utils.py
"""
import re
import sys
import unicodedata

# Greek choice labels
LABELS = ["Α.", "Β.", "Γ.", "Δ."]

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

class GreekMultiChoiceRegexFilter:
    """
    Adapted from MultiChoiceRegexFilter to handle Greek MMLU specificities.
    """

    def __init__(
        self,
        regex_pattern: str = r"#### (\-?[0-9\.\,]+)",
        group_select=0,
        fallback: str = "[invalid]",
        ignore_case=False,
        ignore_punctuation=False,
        regexes_to_ignore=None,
    ) -> None:
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        """
        self.regex_pattern = regex_pattern
        self.group_select = group_select
        self.fallback = fallback
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore
        self.regex = re.compile(regex_pattern)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match

        punct_tbl = dict.fromkeys(
            i
            for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith("P")
        )

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, "", st)

            if self.ignore_case:
                st = st.lower()

            if self.ignore_punctuation:
                # https://stackoverflow.com/a/266162
                st = st.translate(punct_tbl)
            return st

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            
            # Greek mapping for labels
            # We map 0 -> Α, 1 -> Β, etc.
            greek_map = ["Α", "Β", "Γ", "Δ"]
            
            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            choices = doc["choices"]
            for i, c in enumerate(choices):
                m = filter_ignores(c.strip())
                current_alpha = greek_map[i] if i < len(greek_map) else str(i)
                
                # Match exact choice text
                fallback_regexes.append(f"{re.escape(m)}")
                choice_to_alpha[m] = current_alpha

                # Match letter
                without_paren_fallback_regexes.append(current_alpha)
                without_paren_to_target[current_alpha] = current_alpha
                # Also allow matching "A" for "Α" etc in case model outputs Latin
                latin_map = {"Α": "A", "Β": "B", "Γ": "C", "Δ": "D"}
                if current_alpha in latin_map:
                    latin_char = latin_map[current_alpha]
                    without_paren_fallback_regexes.append(latin_char)
                    without_paren_to_target[latin_char] = current_alpha

            fallback_regex = re.compile("|".join(fallback_regexes))
            
            # Regex to look for the letter. 
            # We relax it to allow "Answer: A" or "A." or just "A" at end of text ideally
            # But adhering to the structure:
            without_paren_fallback_regex_str = "|".join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(
                rf"(?:^|\s|[^\w])({without_paren_fallback_regex_str})(?:$|\s|[^\w])"
            )

            filtered = []
            for resp in r:
                # 1. Try strict regex pattern (e.g. "Answer: A")
                match = find_match(self.regex, resp)
                if not match:
                    # 2. Try matching the full text of the choice
                    match = find_match(
                        fallback_regex, filter_ignores(resp), choice_to_alpha
                    )
                    if not match:
                        # 3. Try matching just the letter
                        # We use findall and take the *LAST* one for CoT usually, 
                        # but MMLU logic uses group_select (default 0? No, wait)
                        # MMLU MultiChoiceRegexFilter uses group_select=0 by default? 
                        # In the YAML it says group_select: -1 (take last)
                        
                        # We need to manually use the regex with findall
                        matches = without_paren_fallback_regex.findall(resp)
                        if matches:
                            # If group_select is -1, take the last one
                            idx = self.group_select
                            if idx < 0:
                                idx = len(matches) + idx
                            if 0 <= idx < len(matches):
                                match = matches[idx]
                                if match in without_paren_to_target:
                                    match = without_paren_to_target[match]
                            else:
                                match = None
                        else:
                             match = None

                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)

        return filtered_resps


PROMPT = "Αυτό είναι μια ερώτηση {}. Ερώτηση: {} {} Ας σκεφτούμε βήμα προς βήμα και ας επιλέξουμε τη σωστή απάντηση.\n"

def doc_to_text(doc):
    """
    Format the question with choices for Greek MMLU.
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


def doc_to_target(doc):
    """
    Convert answer index to Greek letter string.
    """
    num_choices = len(doc["choices"])
    valid_labels = [LABELS[i][0] for i in range(num_choices)]
    return valid_labels[doc["answer"]]


def process_results(doc, results):
    """
    Process results using the GreekMultiChoiceRegexFilter.
    """
    # Configure the filter
    # We want to match explicitly "Απάντηση: X" or similar first
    # Then fallback to last valid letter
    
    # Pattern 1: Explicit answer
    # Matches: "Απάντηση: A", "Answer: B", "Η σωστή απάντηση είναι Γ"
    # Note: RegexFilter expects finding the GROUP
    explicit_pattern = r"(?:απάντηση|answer|επιλογή)[:\s]+([ΑΒΓΔABCD])"
    
    filter_obj = GreekMultiChoiceRegexFilter(
        regex_pattern=explicit_pattern,
        group_select=-1, # Take the last explicit answer if multiple
        ignore_case=True,
        ignore_punctuation=True
    )
    
    # Mocking the list structure expected by apply
    # apply expects [[resp1], [resp2]] for multiple docs
    # Here we have 1 doc, and results is [resp]
    resps_list = [[results[0]]]
    docs_list = [doc]
    
    filtered_results = filter_obj.apply(resps_list, docs_list)
    
    # filtered_results is [[match]]
    pred = filtered_results[0][0]
    
    gold = doc_to_target(doc) # e.g. "Α"
    
    if pred == "[invalid]" or pred is None:
        return {"acc": 0.0}
        
    return {"acc": float(pred == gold)}
