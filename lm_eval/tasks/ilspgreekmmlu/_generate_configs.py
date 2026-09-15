"""
Take in a YAML, and output all other splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


# The 57 MMLU subjects, as machine-translated into Greek by ILSP in
# https://huggingface.co/datasets/ilsp/mmlu_greek (the `all` config is skipped).
SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_template_yaml")
    parser.add_argument("--save_prefix_path", default="ilspgreekmmlu")
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    for subject in tqdm(SUBJECTS):
        task_name = (
            f"{args.save_prefix_path}_{args.task_prefix}_{subject}"
            if args.task_prefix != ""
            else f"{args.save_prefix_path}_{subject}"
        )
        yaml_dict = {
            "include": base_yaml_name,
            "task": task_name,
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                sort_keys=False,
            )

    # write group config out
    group_yaml_dict = {
        "group": args.save_prefix_path,
        "task": [
            (
                f"{args.save_prefix_path}_{args.task_prefix}_{subject}"
                if args.task_prefix != ""
                else f"{args.save_prefix_path}_{subject}"
            )
            for subject in SUBJECTS
        ],
        "aggregate_metric_list": [
            {"metric": "acc", "aggregation": "mean", "weight_by_size": True},
        ],
        "metadata": {"version": 1.0},
    }

    file_save_path = "_" + args.save_prefix_path + ".yaml"
    with open(file_save_path, "w", encoding="utf-8") as group_yaml_file:
        yaml.dump(
            group_yaml_dict,
            group_yaml_file,
            width=float("inf"),
            allow_unicode=True,
            sort_keys=False,
        )

# python _generate_configs.py --base_yaml_path _default_template_yaml --save_prefix_path ilspgreekmmlu
