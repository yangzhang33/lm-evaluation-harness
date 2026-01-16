import os
from datasets import load_from_disk

# Check if environment variable is set
if os.environ.get("HF_DATASETS_OFFLINE") != "1":
    print("Warning: HF_DATASETS_OFFLINE is not set to 1. This test might not be valid for strict offline verification.")
else:
    print("Environment is set to OFFLINE mode (HF_DATASETS_OFFLINE=1).")

local_path = "/home/mersin-konomi/model_eval/lm-evaluation-harness/greekmmlu_offline"
print(f"Attempting to load dataset from: {local_path}")

try:
    # Attempt to load from the local path
    dataset = load_from_disk(local_path)
    print("\nSuccess! Dataset loaded correctly from disk.")
    print("------------------------------------------------")
    print(f"Dataset Structure: {dataset}")
    print("------------------------------------------------")
    
    # Simple check to see if we can access data
    if 'test' in dataset:
        print(f"Test split size: {len(dataset['test'])}")
        print(f"First example question: {dataset['test'][0].get('question', 'N/A')}")
    
except Exception as e:
    print(f"\nFAILED to load dataset: {e}")
    exit(1)
