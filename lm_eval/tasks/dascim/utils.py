from typing import List
from datasets import load_dataset, DatasetDict, concatenate_datasets, get_dataset_config_names

# Letter labels for multiple choice questions
letters = ["A", "B", "C", "D"]


def custom_dataset(**kwargs):
    """Load ALL configurations from the dataset and concatenate them.
    This loads all 66 configurations (~15,000+ questions) across all subjects.
    Also fixes answer indexing issues (some questions have 1-indexed answers).
    """
    # Get all available configurations
    configs = get_dataset_config_names('mkonomi/greek_mmlu_configs', trust_remote_code=True)
    print(f"Loading {len(configs)} configurations...")
    
    all_test_splits = []
    fixed_count = 0
    
    for cfg in configs:
        try:
            # Load only test split for each configuration
            ds = load_dataset('mkonomi/greek_mmlu_configs', cfg, split='test', trust_remote_code=True)
            
            # Fix answer indexing: convert 1-indexed to 0-indexed where needed
            def fix_answer(example):
                nonlocal fixed_count
                n_choices = len(example['choices'])
                # If answer is out of range (1-indexed), convert to 0-indexed
                if example['answer'] >= n_choices:
                    fixed_count += 1
                    example['answer'] = example['answer'] - 1
                return example
            
            ds = ds.map(fix_answer)
            all_test_splits.append(ds)
        except Exception as e:
            print(f"Warning: Could not load {cfg}: {e}")
            continue
    
    # Concatenate all test splits
    combined_test = concatenate_datasets(all_test_splits)
    print(f"Total questions loaded: {len(combined_test)}")
    print(f"Fixed {fixed_count} questions with incorrect answer indexing")
    
    # Return as DatasetDict with test split only
    return DatasetDict({
        'test': combined_test
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

