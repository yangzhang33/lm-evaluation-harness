
# noqa
"""
Generates YAMLs for Greek MMLU Split API task.
"""

import argparse
import logging
import os
import yaml
import json
from tqdm import tqdm
from collections import defaultdict

# Create logger object
eval_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUBJECTS = {
    "Abstract_Algebra": "STEM",
    "Accounting": "Other",
    "Agriculture": "STEM",
    "Anatomy": "STEM",
    "Ancient_Greek_Language": "Social_Sciences",
    "Art": "Humanities",
    "Astronomy": "STEM",
    "Biology": "STEM",
    "Business_Ethics": "Social_Sciences",
    "Chemistry": "STEM",
    "Civil_Engineering": "STEM",
    "Clinical_Knowledge": "STEM",
    "Commonsense": "Other",
    "Computer_Science": "STEM",
    "Computer_Networks_&_Security": "STEM",
    "Driving_Rules": "Other",
    "Economics": "Social_Sciences",
    "Education": "Social_Sciences",
    "Electrical Engineering": "STEM",
    "European_History": "Humanities",
    "Formal_Logic": "Humanities",
    "Geography": "Social_Sciences",
    "General_Knowledge": "Other",
    "Global_Facts": "Other",
    "Government_and_Politics": "Social_Sciences",
    "Greek_Traditions": "Social_Sciences",
    "Greek_History": "Humanities",
    "Greek_Mythology": "Humanities",
    "Greek_Literature": "Humanities",
    "Human_Aging": "Other",
    "Human_Sexuality": "Social_Sciences",
    "International_Law": "Humanities",
    "Law": "Humanities",
    "Logical_Fallacies": "Humanities",
    "Machine_Learning": "STEM",
    "Management": "Social_Sciences",
    "Marketing": "Social_Sciences",
    "Mathematics": "STEM",
    "Medicine": "STEM",
    "Modern_Greek_Language": "Social_Sciences",
    "Moral_Disputes": "Humanities",
    "Moral_Scenarios": "Humanities",
    "Philosophy": "Humanities",
    "Physics": "STEM",
    "Prehistory": "Humanities",
    "Psychology": "Social_Sciences",
    "Public_Relations": "Social_Sciences",
    "Security_Studies": "Social_Sciences",
    "Sociology": "Social_Sciences",
    "Statistics": "STEM",
    "Virology": "Other",
    "World_History": "Humanities",
    "World_Religions": "Humanities",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="greekmmlu_split")
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    parser.add_argument("--dataset_path", default="mkonomi/GreekMMLU-Public-Split")
    parser.add_argument("--process_data", action="store_true", help="Included for compatibility, but data splitting is done separately.")
    parser.add_argument("--input_data_dir", default=None)
    parser.add_argument("--output_data_dir", default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Create default directory relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Save directly in the script dir or a subdirectory? 
    # Usually lm-eval expects them in the task directory.
    # The user might want them in a 'greekmmlu_split_api' folder or similar.
    # Let's save them in the current directory for now, or a folder.
    # Assuming this script is running in 'lm_eval/tasks/greekmmlu_api/'
    
    # We will generate configs for each Subject_Nchoice combination.
    # Since we don't have the data locally to check presence, 
    # we will assume all subjects have 2, 3, 4 partitions OR 
    # we can fetch the dataset info. But simpler is to generate all potential YAMLs 
    # OR better: The user wants to use this config for the "greek api duplicate".
    # We should probably fetch the available configs from the HF dataset to be accurate.
    
    # Let's try to fetch configs from HF
    try:
        from datasets import get_dataset_config_names
        available_configs = get_dataset_config_names(args.dataset_path)
        eval_logger.info(f"Retrieved {len(available_configs)} configs from {args.dataset_path}")
    except Exception as e:
        eval_logger.warning(f"Could not fetch configs from HF: {e}. Falling back to generating all likely combinations.")
        available_configs = []
        # Fallback generation logic if needed
        # But DASCIM splits are strictly what exists.
    
    # Copy base template
    base_yaml_name = "_default_greekmmlu_api_split_template_yaml"
    base_yaml_path = os.path.join(script_dir, base_yaml_name)
    
    # We need a base template that uses the new dataset. 
    # It should look like the original but with dataset_path parameterizable or set in individual yamls.
    # We will read args.base_yaml_path and modify it or use it as is if it's generic enough.
    
    try:
        with open(args.base_yaml_path, 'r') as f:
            base_config = yaml.safe_load(f)
    except:
        # Create a default base if path invalid
        base_config = {
            "dataset_path": args.dataset_path,
            "test_split": "test",
            "fewshot_split": "dev",
            "fewshot_config": { "sampler": "first_n" },
            "output_type": "generate_until",
            "generation_kwargs": {
                "until": "\n\n", # Adjusted for API often
                "max_gen_toks": 32,
                "do_sample": False,
                "temperature": 0.0
            },
            "doc_to_text": "!function utils.doc_to_text",
            "doc_to_target": "!function utils.doc_to_target",
            "process_results": "!function utils.process_results",
            "metric_list": [
                { "metric": "acc", "aggregation": "mean", "higher_is_better": True }
            ],
            "metadata": { "version": 2.0 }
        }
        # Be careful with !function, safe_load parses as string or error.
        # harness uses custom loader. We should just copy the file if possible to preserve tags.

    import shutil
    try:
        shutil.copy2(args.base_yaml_path, base_yaml_path)
    except shutil.SameFileError:
        pass


    ALL_GENERATED_TASKS = []
    
    # Iterate over available configs from HF
    for config_name in tqdm(available_configs):
        # Config name format: {Subject}_{N}choice or {Subject}_other
        # We need to extract Subject and choice count.
        
        # Heuristic: split by last underscore
        parts = config_name.rsplit('_', 1)
        if len(parts) != 2:
            continue
            
        subject, suffix = parts
        # Validation: suffix should be 2choice, 3choice, 4choice, other
        
        # Get category
        category = SUBJECTS.get(subject, "Other")
        category_lower = category.lower()
        
        # Task name
        task_name = f"greekmmlu_split_{subject}_{suffix}"
        if args.task_prefix:
            task_name = f"greekmmlu_split_{args.task_prefix}_{subject}_{suffix}"
            
        dataset_name = config_name
        
        # Generate YAML
        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"greekmmlu_split_{category_lower}_tasks", # Group by category
            "task": task_name,
            "task_alias": f"{subject} ({suffix})",
            "dataset_path": args.dataset_path, 
            "dataset_name": dataset_name,
            # Description could be specific if needed
            "description": f"Multiple choice questions for {subject} with {suffix} answers.\\n\\n"
        }
        
        yaml_filename = f"{task_name}.yaml"
        yaml_path = os.path.join(script_dir, yaml_filename)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_dict, f, default_flow_style=False)
            
        ALL_GENERATED_TASKS.append(task_name)
    
    # Create Group Configs
    # Main group
    group_name = "greekmmlu_split"
    if args.task_prefix:
        group_name += f"_{args.task_prefix}"
        
    with open(os.path.join(script_dir, f"_{group_name}.yaml"), 'w') as f:
        yaml.dump({
            "group": group_name,
            "task": ALL_GENERATED_TASKS,
             "aggregate_metric_list": [
                    {
                        "metric": "acc",
                        "weight_by_size": True
                    }
                ],
            "metadata": {"version": 2.0}
        }, f)
        
    eval_logger.info(f"Generated {len(ALL_GENERATED_TASKS)} task YAMLs.")

