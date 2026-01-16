from typing import List
from datasets import load_dataset, DatasetDict, concatenate_datasets, get_dataset_config_names

# Letter labels for multiple choice questions
letters = ["A", "B", "C", "D"]


def custom_dataset(dataset_name=None, **kwargs):
    """Load dataset with specific configuration for test, but ALL configs for few-shot.
    This allows cross-subject few-shot examples (e.g., Physics examples in Medicine task).
    """
    if dataset_name is None:
        raise ValueError("dataset_name must be provided in metadata")
    
    # Load specific config for test examples (this subject only)
    test_split = load_dataset(
        'mkonomi/greek_mmlu_configs', 
        dataset_name, 
        split='test', 
        trust_remote_code=True
    )
    
    # Load ALL configs for few-shot examples (cross-subject)
    all_configs = get_dataset_config_names('mkonomi/greek_mmlu_configs', trust_remote_code=True)
    fewshot_splits = []
    
    for cfg in all_configs:
        try:
            ds = load_dataset(
                'mkonomi/greek_mmlu_configs', 
                cfg, 
                split='test', 
                trust_remote_code=True
            )
            fewshot_splits.append(ds)
        except Exception as e:
            # Skip configs that fail to load
            continue
    
    # Concatenate all configs for few-shot pool
    combined_fewshot = concatenate_datasets(fewshot_splits) if fewshot_splits else test_split
    
    # Return DatasetDict with test (subject-specific) and fewshot (all subjects)
    return DatasetDict({
        'test': test_split,           # Subject-specific test examples
        'fewshot': combined_fewshot   # All subjects for few-shot examples
    })


def doc_to_text(doc) -> str:
    """
    Dynamically formats a document with variable number of choices.
    
    Args:
        doc (dict): A document with 'question' and 'choices' fields.
                   'choices' should be a list with 2, 3, or 4 items.
    
    Returns:
        str: Formatted string with question and answer choices.
    """
    choices = doc["choices"]
    num_choices = len(choices)
    
    if num_choices < 2 or num_choices > 4:
        raise ValueError(f"Invalid number of choices: {num_choices}. Expected 2-4.")
    
    # Get the appropriate letter labels
    choice_letters = letters[:num_choices]
    
    # Format: Question + A. choice1\nB. choice2\n...
    formatted_choices = "\n".join(
        [f"{letter}. {choices[i]}" for i, letter in enumerate(choice_letters)]
    )
    
    return f"{doc['question'].strip()}\n{formatted_choices}\nAnswer:"


def doc_to_choice(doc) -> List[str]:
    """
    Dynamically returns the answer choice labels based on number of choices.
    
    Args:
        doc (dict): A document with 'choices' field.
    
    Returns:
        list: List of choice labels (e.g., ["A", "B"] for 2 choices, 
              ["A", "B", "C", "D"] for 4 choices).
    """
    choices = doc["choices"]
    num_choices = len(choices)
    
    if num_choices < 2 or num_choices > 4:
        raise ValueError(f"Invalid number of choices: {num_choices}. Expected 2-4.")
    
    return letters[:num_choices]

