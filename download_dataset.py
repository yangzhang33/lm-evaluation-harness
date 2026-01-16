from datasets import load_dataset
import os

# Ορίζουμε το path όπου θα αποθηκευτεί το dataset μέσα στο τρέχον project
local_path = "/home/mersin-konomi/model_eval/lm-evaluation-harness/greekmmlu_offline"

print(f"Ξεκινά η λήψη του dataset στο: {local_path}")

# 1. Κατεβάζουμε το dataset από το HF
dataset = load_dataset("mkonomi/GreekMMLU-Public", "All")

# 2. Το αποθηκεύουμε τοπικά στον δίσκο
dataset.save_to_disk(local_path)

print(f"Το dataset κατέβηκε επιτυχώς και αποθηκεύτηκε στο: {local_path}!")
