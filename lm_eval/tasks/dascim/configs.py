# noqa
"""
Generates YAMLs for LM-Evaluation-Harness.
FEATURES:
1. AUTOMATICALLY SPLITS data by choice count (2, 3, 4 choices) FIRST.
2. Then splits by subject within each choice count.
3. Does NOT modify the questions (No shuffling/decontamination).
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
    "Abstract Algebra": "STEM",
    "Accounting": "Other",
    "Agriculture": "STEM",
    "Agroculture": "STEM",
    "Anatomy": "STEM",
    "Ancient Greek Language": "Social_Sciences",
    "Art": "Humanities",
    "Astronomy": "STEM",
    "Biology": "STEM",
    "Business Ethics": "Social_Sciences",
    "Chemistry": "STEM",
    "Civil Engineering": "STEM",
    "Clinical Knowledge": "STEM",
    "Commonsense": "Other",
    "Computer Science": "STEM",
    "Computer Networks & Security": "STEM",
    "Driving Rules": "Other",
    "Economics": "Social_Sciences",
    "Education": "Social_Sciences",
    "Electrical Engineering": "STEM",
    "European History": "Humanities",
    "Formal Logic": "Humanities",
    "Geography": "Social_Sciences",
    "General Knowledge": "Other",
    "Global Facts": "Other",
    "Government and Politics": "Social_Sciences",
    "Greek Traditions": "Social_Sciences",
    "Greek History": "Humanities",
    "Greek Mythology": "Humanities",
    "Greek Literature": "Humanities",
    "Human Aging": "Other",
    "Human Sexuality": "Social_Sciences",
    "International Law": "Humanities",
    "Law": "Humanities",
    "Logical Fallacies": "Humanities",
    "Machine Learning": "STEM",
    "Management": "Social_Sciences",
    "Marketing": "Social_Sciences",
    "Mathematics": "STEM",
    "Medicine": "STEM",
    "Modern Greek Language": "Social_Sciences",
    "Moral Disputes": "Humanities",
    "Moral Scenarios": "Humanities",
    "Philosophy": "Humanities",
    "Physics": "STEM",
    "Prehistory": "Humanities",
    "Psychology": "Social_Sciences",
    "Public Relations": "Social_Sciences",
    "Security Studies": "Social_Sciences",
    "Sociology": "Social_Sciences",
    "Statistics": "STEM",
    "Virology": "Other",
    "World History": "Humanities",
    "World Religion": "Humanities",
    "World Religions": "Humanities",
}

# ==========================================
#  DATA SPLITTING FUNCTION
# ==========================================

def split_by_choice_then_subject(input_dir, output_dir):
    """
    Reads all JSON files from input_dir, splits by choice count first,
    then by subject within each choice count.
    Returns: dict mapping (choice_count, subject) -> output_file_path
    """
    # Structure: {choice_count: {subject: [items]}}
    data_structure = defaultdict(lambda: defaultdict(list))
    
    # Read all JSON files
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(input_dir, filename)
        eval_logger.info(f"Reading {filename}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    
                    # Get choice count and subject
                    choice_count = len(item.get("choices", []))
                    subject = item.get("subject", "Unknown")
                    
                    # Categorize by choice count first, then subject
                    if choice_count in [2, 3, 4]:
                        data_structure[choice_count][subject].append(item)
                    else:
                        data_structure["other"][subject].append(item)
                        
                except json.JSONDecodeError:
                    continue
    
    # Save files and record paths
    created_files = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for choice_count, subjects_dict in data_structure.items():
        for subject, items in subjects_dict.items():
            if not items:
                continue
            
            # Create safe filename (replace spaces with underscores)
            safe_subject = subject.replace(' ', '_').replace('&', 'and')
            suffix = f"{choice_count}choice" if choice_count != "other" else "other"
            out_filename = f"{safe_subject}_{suffix}.json"
            out_path = os.path.join(output_dir, out_filename)
            
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
            
            created_files[(choice_count, subject)] = out_path
            eval_logger.info(f"  -> Created {suffix} split for {subject} ({len(items)} questions)")
    
    return created_files

# ==========================================
#  MAIN SCRIPT
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="dascim")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")

    # Process Data Args
    parser.add_argument("--process_data", action="store_true", help="Turn this on to actually split the JSON files.")
    parser.add_argument("--input_data_dir", default=None)
    parser.add_argument("--output_data_dir", default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    cot_file = {}
    if args.cot_prompt_path is not None:
        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    ALL_GENERATED_TASKS = []
    # Track tasks by category: {category: [task_names]}
    tasks_by_category = defaultdict(list)
    category_group_names = []  # Will hold category group names if generated
    
    # Create default directory relative to this script's location (in dascim folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, "default")
    os.makedirs(default_dir, exist_ok=True)

    # ---------------------------------------------------------
    # GENERATE BASE TEMPLATE FILES (Master base + choice-specific)
    # ---------------------------------------------------------
    # Copy the master base template directly to preserve !function tags
    # (The harness's YAML loader handles !function, but standard yaml.load doesn't)
    master_base_filename = "_default_template_yaml"
    master_base_path = os.path.join(default_dir, master_base_filename)
    
    eval_logger.info(f"Creating master base template: {master_base_path}")
    import shutil
    shutil.copy2(args.base_yaml_path, master_base_path)
    
    # Copy utils.py to default directory if it exists in the dascim task directory
    # This ensures !function utils.* references work correctly
    dascim_utils_path = os.path.join(script_dir, "utils.py")
    default_utils_path = os.path.join(default_dir, "utils.py")
    if os.path.exists(dascim_utils_path):
        eval_logger.info(f"Copying utils.py to default directory: {default_utils_path}")
        shutil.copy2(dascim_utils_path, default_utils_path)
    
    # Generate choice-specific base files that include the master base
    # Note: We no longer use num_choices (not a valid TaskConfig parameter)
    # Instead, doc_to_text and doc_to_choice functions handle variable choices dynamically
    base_yaml_templates = {}
    for choice_count in [2, 3, 4]:
        base_filename = f"base_dascim_{choice_count}choice.yaml"
        base_path = os.path.join(default_dir, base_filename)
        
        # These files exist mainly for organizational purposes
        # The actual choice handling is done by utils.py functions
        base_dict = {
            "include": master_base_filename
        }
        
        eval_logger.info(f"Creating base template for {choice_count} choices: {base_path}")
        with open(base_path, "w", encoding="utf-8") as f:
            yaml.dump(base_dict, f, allow_unicode=True, default_style='"', indent=2)
        
        base_yaml_templates[choice_count] = base_path
    
    # For "other" choice counts, use the master base directly
    base_yaml_templates["other"] = master_base_path

    # ---------------------------------------------------------
    # SPLIT DATA & GENERATE YAMLs
    # ---------------------------------------------------------
    if args.process_data:
        if not args.input_data_dir or not args.output_data_dir:
            raise ValueError("Must provide --input_data_dir and --output_data_dir")

        # Split by choice count first, then by subject
        split_files = split_by_choice_then_subject(args.input_data_dir, args.output_data_dir)

        if not split_files:
            eval_logger.warning("No data found, exiting.")
        else:
            # Generate YAML for each (choice_count, subject) combination
            for (choice_count, subject), file_path in tqdm(split_files.items()):
                # Get category for this subject
                category = SUBJECTS.get(subject, "Other")
                # Normalize category name (STEM -> stem, Social_Sciences -> social_sciences, etc.)
                category_normalized = category.lower()
                
                # Create description
                description = cot_file.get(subject, f"The following are multiple choice questions (with answers) about {subject}.\\n\\n")
                
                # Create safe names for files and tasks
                safe_subject = subject.replace(' ', '_').replace('&', 'and')
                suffix = f"{choice_count}choice" if choice_count != "other" else "other"
                
                # Get the appropriate base YAML for this choice count
                base_yaml_path = base_yaml_templates.get(choice_count, base_yaml_templates["other"])
                base_yaml_name = os.path.basename(base_yaml_path)
                
                task_name = f"dascim_{args.task_prefix}_{safe_subject}_{suffix}" if args.task_prefix else f"dascim_{safe_subject}_{suffix}"
                # Tag name should have "_tasks" suffix to match MMLU pattern
                tag_name = f"dascim_{args.task_prefix}_{category_normalized}_tasks" if args.task_prefix else f"dascim_{category_normalized}_tasks"

                # Create dataset_name from subject and suffix (e.g., "Art_4choice")
                dataset_name = f"{safe_subject}_{suffix}"
                
                yaml_dict = {
                    "include": base_yaml_name,
                    "tag": tag_name, 
                    "task": task_name,
                    "task_alias": f"{subject} ({suffix})",
                    "dataset_name": dataset_name,
                    "description": description,
                }

                # Save YAML file in default directory
                yaml_filename = f"{args.save_prefix_path}_{safe_subject}_{suffix}.yaml"
                yaml_filepath = os.path.join(default_dir, yaml_filename)
                with open(yaml_filepath, "w", encoding="utf-8") as yf:
                    yaml.dump(yaml_dict, yf, allow_unicode=True, default_style='"')

                ALL_GENERATED_TASKS.append(task_name)
                tasks_by_category[category_normalized].append(task_name)

    # ---------------------------------------------------------
    # GENERATE CATEGORY-LEVEL GROUP FILES
    # ---------------------------------------------------------
    if tasks_by_category:
        all_categories = sorted(tasks_by_category.keys())
        category_group_names.clear()  # Clear previous entries if re-running
        
        for category in all_categories:
            # Create category-level group YAML
            category_group_name = f"dascim_{args.task_prefix}_{category}" if args.task_prefix else f"dascim_{category}"
            category_group_names.append(category_group_name)
            
            # Reference the tag name instead of listing individual tasks (matches MMLU pattern)
            tag_reference = f"dascim_{args.task_prefix}_{category}_tasks" if args.task_prefix else f"dascim_{category}_tasks"
            
            category_yaml_dict = {
                "group": category_group_name,
                "group_alias": category,
                "task": [tag_reference],  # Reference tag instead of listing tasks
                "aggregate_metric_list": [
                    {
                        "metric": "acc",
                        "weight_by_size": True
                    }
                ],
                "metadata": {
                    "version": 2
                }
            }
            
            category_yaml_filename = f"_dascim_{category}.yaml"
            category_yaml_filepath = os.path.join(default_dir, category_yaml_filename)
            
            eval_logger.info(f"Creating category group file: {category_yaml_filepath}")
            with open(category_yaml_filepath, "w", encoding="utf-8") as yf:
                yaml.dump(category_yaml_dict, yf, indent=2, default_flow_style=False)

    # ---------------------------------------------------------
    # FINAL GROUP CONFIG (Main dascim.yaml)
    # ---------------------------------------------------------
    if category_group_names:
        # Use category groups if available
        group_tasks = category_group_names
    else:
        # Fallback to all individual tasks
        group_tasks = ALL_GENERATED_TASKS
    
    # Save main group file in default directory
    main_group_yaml_filepath = os.path.join(default_dir, f"_{args.save_prefix_path}.yaml")
    
    eval_logger.info(f"Saving main group config to {main_group_yaml_filepath}")

    with open(main_group_yaml_filepath, "w", encoding="utf-8") as yf:
        yaml.dump(
            {
                "group": f"dascim_{args.task_prefix}" if args.task_prefix else "dascim",
                "task": group_tasks,
                "aggregate_metric_list": [
                    {
                        "metric": "acc",
                        "weight_by_size": True
                    }
                ],
                "metadata": {
                    "version": 2
                }
            },
            yf,
            indent=4,
            default_flow_style=False,
        )
    
    # Also save the flat group config (for backward compatibility)
    if args.group_prefix != "":
        group_file_path = args.group_prefix + ".yaml"
    else:
        group_file_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving flat group config to {group_file_path}")

    with open(group_file_path, "w", encoding="utf-8") as yf:
        yaml.dump(
            {
                "group": f"dascim_{args.task_prefix}" if args.task_prefix else "dascim",
                "task": ALL_GENERATED_TASKS,
            },
            yf,
            indent=4,
            default_flow_style=False,
        )