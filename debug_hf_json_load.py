
import json
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)

# Slugs from metadata (manually listed or loaded)
slugs = [
    "accounting", "agriculture", "art", "biology", "chemistry", "civil_engineering",
    "clinical_knowledge", "computer_networks_and_security", "computer_science",
    "driving_rules", "economics", "education", "electrical_engineering", 
    "general_knowledge", "geography", "government_and_politics", "greek_history",
    "greek_literature", "greek_mythology", "greek_traditions", "law",
    "management", "mathematics", "medicine", "modern_greek_language",
    "philosophy", "physics", "prehistory", "world_history", "world_religions"
]

for slug in slugs:
    print(f"Testing slug: {slug}")
    data_files = {
        "test": f"https://huggingface.co/datasets/mkonomi/greek_mmlu/resolve/main/data/{slug}_test.jsonl",
        "dev": f"https://huggingface.co/datasets/mkonomi/greek_mmlu/resolve/main/data/{slug}_dev.jsonl",
    }
    try:
        ds = load_dataset("json", data_files=data_files)
        print(f"SUCCESS: {slug}")
    except Exception as e:
        print(f"FAILED: {slug}")
        print(e)
