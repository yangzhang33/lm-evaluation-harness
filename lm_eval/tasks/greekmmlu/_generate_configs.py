"""
Take in a YAML, and output all other splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger(__name__)


SUBJECTS = {
    "abstract_algebra": "Αφηρημένη Άλγεβρα",
    # "all": "Όλα",
    "anatomy": "Ανατομία",
    "astronomy": "Αστρονομία",
    "business_ethics": "Επιχειρηματική Ηθική",
    "clinical_knowledge": "Κλινική Γνώση",
    "college_biology": "Πανεπιστημιακή Βιολογία",
    "college_chemistry": "Πανεπιστημιακή Χημεία",
    "college_computer_science": "Πανεπιστημιακή Επιστήμη Υπολογιστών",
    "college_mathematics": "Πανεπιστημιακά Μαθηματικά",
    "college_medicine": "Πανεπιστημιακή Ιατρική",
    "college_physics": "Πανεπιστημιακή Φυσική",
    "computer_security": "Ασφάλεια Υπολογιστών",
    "conceptual_physics": "Εννοιολογική Φυσική",
    "econometrics": "Οικονομετρία",
    "electrical_engineering": "Ηλεκτρολογική Μηχανική",
    "elementary_mathematics": "Στοιχειώδη Μαθηματικά",
    "formal_logic": "Τυπική Λογική",
    "global_facts": "Παγκόσμια Γεγονότα",
    "high_school_biology": "Λυκειακή Βιολογία",
    "high_school_chemistry": "Λυκειακή Χημεία",
    "high_school_computer_science": "Λυκειακή Επιστήμη Υπολογιστών",
    "high_school_european_history": "Λυκειακή Ευρωπαϊκή Ιστορία",
    "high_school_geography": "Λυκειακή Γεωγραφία",
    "high_school_government_and_politics": "Λυκειακή Κυβέρνηση και Πολιτική",
    "high_school_macroeconomics": "Λυκειακή Μακροοικονομία",
    "high_school_mathematics": "Λυκειακά Μαθηματικά",
    "high_school_microeconomics": "Λυκειακή Μικροοικονομία",
    "high_school_physics": "Λυκειακή Φυσική",
    "high_school_psychology": "Λυκειακή Ψυχολογία",
    "high_school_statistics": "Λυκειακή Στατιστική",
    "high_school_us_history": "Λυκειακή Ιστορία ΗΠΑ",
    "high_school_world_history": "Λυκειακή Παγκόσμια Ιστορία",
    "human_aging": "Γήρανση του Ανθρώπου",
    "human_sexuality": "Ανθρώπινη Σεξουαλικότητα",
    "international_law": "Διεθνές Δίκαιο",
    "jurisprudence": "Νομολογία",
    "logical_fallacies": "Λογικά Σφάλματα",
    "machine_learning": "Μηχανική Μάθηση",
    "management": "Διοίκηση",
    "marketing": "Μάρκετινγκ",
    "medical_genetics": "Ιατρική Γενετική",
    "miscellaneous": "Διάφορα",
    "moral_disputes": "Ηθικές Διαμάχες",
    "moral_scenarios": "Ηθικά Σενάρια",
    "nutrition": "Διατροφή",
    "philosophy": "Φιλοσοφία",
    "prehistory": "Προϊστορία",
    "professional_accounting": "Επαγγελματική Λογιστική",
    "professional_law": "Επαγγελματικό Δίκαιο",
    "professional_medicine": "Επαγγελματική Ιατρική",
    "professional_psychology": "Επαγγελματική Ψυχολογία",
    "public_relations": "Δημόσιες Σχέσεις",
    "security_studies": "Σπουδές Ασφαλείας",
    "sociology": "Κοινωνιολογία",
    "us_foreign_policy": "Εξωτερική Πολιτική ΗΠΑ",
    "virology": "Ιολογία",
    "world_religions": "Παγκόσμιες Θρησκείες"
}



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="greekmmlu")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)
    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    for subject_eng, subject_el in tqdm(SUBJECTS.items()):
        if args.cot_prompt_path is not None:
            description = cot_file[subject_eng]
        else:
            description = (
                f"Ακολουθεί μια ερώτηση πολλαπλής επιλογής σχετικά με το {subject_el}, παρακαλώ δώστε απευθείας τη σωστή απάντηση.\n\n"
            )

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"greekmmlu_{args.task_prefix}_{subject_eng}"
            if args.task_prefix != ""
            else f"greekmmlu_{subject_eng}",
            "dataset_name": subject_eng,
            "description": description,
        }
        print(args.save_prefix_path)
        file_save_path = args.save_prefix_path + f"_{subject_eng}.yaml"
        print(file_save_path)
        eval_logger.info(f"Saving yaml for subset {subject_eng} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

    # write group config out

    group_yaml_dict = {
        "group": "greekmmlu",
        "task": [
            (
                f"greekmmlu_{args.task_prefix}_{subject_eng}"
                if args.task_prefix != ""
                else f"greekmmlu_{subject_eng}"
            )
            for subject_eng in SUBJECTS.keys()
        ],
        "aggregate_metric_list": [
            {"metric": "acc", "aggregation": "mean", "weight_by_size": True},
            {"metric": "acc_norm", "aggregation": "mean", "weight_by_size": True},
        ],
        "metadata": {"version": 0.0},
    }

    file_save_path = "_" + args.save_prefix_path + ".yaml"

    with open(file_save_path, "w", encoding="utf-8") as group_yaml_file:
        yaml.dump(
            group_yaml_dict,
            group_yaml_file,
            width=float("inf"),
            allow_unicode=True,
            default_style='"',
        )
# python _generate_configs.py --base_yaml_path _default_template_yaml --save_prefix_path greekmmlu