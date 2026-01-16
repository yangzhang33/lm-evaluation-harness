"""
Generate Greek MMLU Private task configurations from mkonomi/test_full.
Creates individual YAML files for each subject and category group files.
"""

import argparse
import logging
import os
import yaml
from tqdm import tqdm

eval_logger = logging.getLogger(__name__)

# Subject to category mapping
SUBJECTS = {
    # Humanities
    "Art": "humanities",
    "Greek History": "humanities",
    "Greek Literature": "humanities",
    "Greek Mythology": "humanities",
    "Greek Traditions": "humanities",
    "Prehistory": "humanities",
    "World History": "humanities",
    "World Religions": "humanities",
    
    # Social Sciences
    "Accounting": "social_sciences",
    "Economics": "social_sciences",
    "Education": "social_sciences",
    "Geography": "social_sciences",
    "Government and Politics": "social_sciences",
    "Law": "social_sciences",
    "Management": "social_sciences",
    "Modern Greek Language": "social_sciences",
    
    # STEM
    "Agriculture": "stem",
    "Biology": "stem",
    "Chemistry": "stem",
    "Civil Engineering": "stem",
    "Clinical Knowledge": "stem",
    "Computer Networks & Security": "stem",
    "Computer Science": "stem",
    "Electrical Engineering": "stem",
    "Mathematics": "stem",
    "Medicine": "stem",
    "Physics": "stem",
    
    # Other
    "Driving Rules": "other",
    "General Knowledge": "other",
}

# Subjects WITH level splits
SUBJECTS_WITH_LEVELS = {
    "Agriculture": ["Professional", "University"],
    "Art": ["Professional", "Secondary_School", "University"],
    "Computer Science": ["Professional", "University"],
    "Economics": ["Professional", "University"],
    "Education": ["Professional", "University"],
    "Geography": ["Primary_School", "Secondary_School"],
    "Government and Politics": ["Primary_School", "Secondary_School"],
    "Greek History": ["Primary_School", "Professional", "Secondary_School"],
    "Management": ["Professional", "University"],
    "Medicine": ["Professional", "University"],
    "Modern Greek Language": ["Primary_School", "Secondary_School"],
    "Physics": ["Primary_School", "Professional", "University"],
}

# Subjects WITHOUT level splits
SUBJECTS_WITHOUT_LEVELS = [
    "Accounting", "Biology", "Chemistry", "Civil Engineering", "Clinical Knowledge",
    "Computer Networks & Security", "Driving Rules", "Electrical Engineering",
    "General Knowledge", "Greek Literature", "Greek Mythology", "Greek Traditions",
    "Law", "Mathematics", "Prehistory", "World History", "World Religions"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_greekmmlu_private_template_yaml")
    parser.add_argument("--save_prefix_path", default="greekmmlu_private")
    return parser.parse_args()


def normalize_subject_name(subject):
    """Convert subject name to valid filename."""
    return subject.lower().replace(" ", "_").replace("&", "and")


if __name__ == "__main__":
    args = parse_args()

    # Get filename of base_yaml so we can `"include":` it in our "other" YAMLs
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    eval_logger.info(f"Generating configs for Greek MMLU Private...")

    # 0. Fetch available configs from Hugging Face
    print("\n=== Fetching Available Configs ===")
    try:
        from datasets import get_dataset_config_names
        available_configs = get_dataset_config_names("mkonomi/test_full")
        print(f"Found {len(available_configs)} available configs: {available_configs}")
    except Exception as e:
        print(f"Error fetching configs: {e}")
        available_configs = []

    ALL_CATEGORIES = []
    all_tasks = []  # Track all tasks for main benchmark file


    # 1. Generate plain subject configs
    print("\n=== Generating Plain Subject Configs ===")
    for subject in tqdm(SUBJECTS_WITHOUT_LEVELS, desc="Plain subject configs"):
        # Check if dataset_name exists in available_configs
        hf_config_name = subject.replace(" ", "_").replace("&", "and")
        if hf_config_name not in available_configs:
            # print(f"  [SKIP] {subject} ({hf_config_name}) not in dataset")
            continue

        category = SUBJECTS[subject]
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        normalized_subject = normalize_subject_name(subject)

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"greekmmlu_private_{category}_tasks",
            "task": f"greekmmlu_private_{normalized_subject}",
            "task_alias": subject,
            "dataset_name": hf_config_name,
        }

        file_save_path = f"{args.save_prefix_path}_{normalized_subject}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(yaml_dict, yaml_file, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        all_tasks.append(f"greekmmlu_private_{normalized_subject}")
        # print(f"  ✓ {file_save_path}")

    # 2. Generate subject+level configs
    print("\n=== Generating Subject+Level Configs ===")
    for subject, levels in tqdm(SUBJECTS_WITH_LEVELS.items(), desc="Subject+level configs"):
        category = SUBJECTS[subject]
        
        normalized_subject = normalize_subject_name(subject)
        
        subject_has_tasks = False
        for level in levels:
            # Config name: Subject_Level (e.g., Physics_Professional)
            hf_config_name = f"{subject.replace(' ', '_').replace('&', 'and')}_{level}"
            
            if hf_config_name not in available_configs:
                # print(f"  [SKIP] {subject} {level} ({hf_config_name}) not in dataset")
                continue

            # Add category if we have at least one task
            if category not in ALL_CATEGORIES:
                ALL_CATEGORIES.append(category)
            subject_has_tasks = True

            normalized_level = level.lower()
            task_name = f"greekmmlu_private_{normalized_subject}_{normalized_level}"
            
            # Alias: Subject_Level e.g. "Physics_Professional"
            task_alias = f"{subject}_{level}"

            yaml_dict = {
                "include": base_yaml_name,
                "tag": f"greekmmlu_private_{category}_tasks",
                "task": task_name,
                "task_alias": task_alias,
                "dataset_name": hf_config_name,
            }

            file_save_path = f"{args.save_prefix_path}_{normalized_subject}_{normalized_level}.yaml"
            with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                yaml.dump(yaml_dict, yaml_file, allow_unicode=True, default_flow_style=False, sort_keys=False)
            
            all_tasks.append(task_name)
            # print(f"  ✓ {file_save_path}")

    # 3. Generate category group files
    print("\n=== Generating Category Group Configs ===")
    for category in ALL_CATEGORIES:
        file_save_path = f"_greekmmlu_private_{category}.yaml"
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"greekmmlu_private_{category}",
                    "group_alias": category,
                    "task": [f"greekmmlu_private_{category}_tasks"],
                    "aggregate_metric_list": [
                        {"metric": "acc", "weight_by_size": True}
                    ],
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )
        print(f"  ✓ {file_save_path}")

    # 4. Generate main benchmark file
    print("\n=== Generating Main Benchmark Config ===")
    greekmmlu_subcategories = [f"greekmmlu_private_{category}" for category in ALL_CATEGORIES]

    file_save_path = f"_{args.save_prefix_path}.yaml"
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": "greekmmlu_private",
                "task": greekmmlu_subcategories,
                "aggregate_metric_list": [
                    {"metric": "acc", "aggregation": "mean", "weight_by_size": True}
                ],
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
    print(f"  ✓ {file_save_path}")

    print(f"\n✅ Generation complete!")
