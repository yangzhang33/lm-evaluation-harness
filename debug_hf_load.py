
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

try:
    print("Attempting to load dataset: mkonomi/greek_mmlu, config: accounting")
    ds = load_dataset("mkonomi/greek_mmlu", "accounting", trust_remote_code=True)
    print("Successfully loaded dataset")
    print(ds)
except Exception as e:
    print(f"Failed to load dataset: {e}")
