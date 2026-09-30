def doc_to_target(doc):
    """The gold is the index where multiple_choice_scores is 1.0."""
    scores = doc["multiple_choice_scores"]
    return max(range(len(scores)), key=scores.__getitem__)
