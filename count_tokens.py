import tiktoken
from datasets import load_dataset
import sys

def count_tokens():
    dataset_name = "mkonomi/GreekMMLU-Public"
    splits = ["test", "validation", "dev"]
    encoding_name = "cl100k_base"
    
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as e:
        print(f"Error loading encoding {encoding_name}: {e}")
        sys.exit(1)

    print(f"Loading dataset: {dataset_name} (Config: All)")
    try:
        dataset = load_dataset(dataset_name, "All")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    total_tokens_all_splits = 0

    print(f"{'Split':<15} | {'Tokens':<15}")
    print("-" * 33)

    for split in splits:
        if split not in dataset:
            print(f"{split:<15} | {'N/A (Not found)':<15}")
            continue

        split_tokens = 0
        data = dataset[split]
        
        for item in data:
            question = item.get("question", "")
            choices = item.get("choices", [])
            
            # Combine question and choices
            # Assuming choices is a list of strings
            text_to_encode = question + " " + " ".join(choices)
            
            tokens = encoding.encode(text_to_encode)
            split_tokens += len(tokens)

        print(f"{split:<15} | {split_tokens:<15,}")
        total_tokens_all_splits += split_tokens

    print("-" * 33)
    print(f"{'Total':<15} | {total_tokens_all_splits:<15,}")

if __name__ == "__main__":
    count_tokens()
