# DASCIM - Greek MMLU Benchmark

Greek multiple-choice question answering benchmark based on [mkonomi/greek_mmlu_configs](https://huggingface.co/datasets/mkonomi/greek_mmlu_configs).

## 📁 Folder Structure

```
dascim/
├── README.md                  # This file
├── dascim_all.yaml            # Single task: ALL questions combined (~15k)
├── utils.py                   # Helper functions for dascim_all
├── configs.py                 # YAML generator script (not used at runtime)
├── __init__.py                # Python package marker
│
└── default/                   # Individual subtasks by subject
    ├── _default_template_yaml # Base configuration template
    ├── utils.py               # Helper functions for subtasks
    │
    ├── _dascim.yaml           # GROUP: All categories combined
    ├── _dascim_stem.yaml      # GROUP: STEM subjects
    ├── _dascim_humanities.yaml      # GROUP: Humanities
    ├── _dascim_social_sciences.yaml # GROUP: Social Sciences
    ├── _dascim_other.yaml           # GROUP: Other subjects
    │
    ├── base_dascim_2choice.yaml  # Base for 2-choice questions
    ├── base_dascim_3choice.yaml  # Base for 3-choice questions
    ├── base_dascim_4choice.yaml  # Base for 4-choice questions
    │
    └── dascim_*.yaml          # 66 individual subtask files
```

## 🚀 How to Run

### Run ALL questions at once (~15,000 questions)
```bash
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=MODEL_NAME" \
    --tasks dascim_all \
    --num_fewshot 0 \
    --device cuda:0
```

### Run by Category Group
```bash
# STEM subjects (Biology, Physics, Chemistry, CS, etc.)
--tasks dascim_stem

# Humanities (History, Philosophy, Law, Literature, etc.)
--tasks dascim_humanities

# Social Sciences (Economics, Psychology, Education, etc.)
--tasks dascim_social_sciences

# Other (General Knowledge, Driving Rules, etc.)
--tasks dascim_other

# All categories combined (via group aggregation)
--tasks dascim
```

### Run Individual Subject
```bash
# Specific subject with choice count
--tasks dascim_Medicine_4choice
--tasks dascim_Physics_4choice
--tasks dascim_Greek_History_4choice
```

## 📊 Available Groups & Tasks

| Group | Description | Example Tasks |
|-------|-------------|---------------|
| `dascim_stem` | Science, Technology, Engineering, Math | Biology, Physics, Chemistry, CS, Medicine |
| `dascim_humanities` | Arts, History, Philosophy | Greek History, Philosophy, Law, Literature |
| `dascim_social_sciences` | Social & Behavioral Sciences | Economics, Psychology, Education, Geography |
| `dascim_other` | Miscellaneous | General Knowledge, Driving Rules |

## 🎯 Few-shot Configuration

**Cross-subject few-shot examples:** Few-shot examples can come from **any subject**, not just the same subject as the test question. This allows high few-shot counts (5, 10, 25) even for subtasks with few questions.

**How it works:**
- **Test examples**: Subject-specific (e.g., only Medicine questions for `dascim_Medicine_4choice`)
- **Few-shot examples**: Pool from ALL subjects (~15,000 questions available)

**Supported few-shot values:**
| num_fewshot | Status |
|-------------|--------|
| 0 | ✅ All tasks |
| 3 | ✅ All tasks |
| 5 | ✅ All tasks (cross-subject) |
| 10 | ✅ All tasks (cross-subject) |
| 25 | ✅ All tasks (cross-subject) |

**Note:** Cross-subject few-shot means examples might be from different subjects (e.g., a Physics example in a Medicine task). This is intentional and allows testing generalization.

## 📁 File Descriptions

### Root Level Files

| File | Description |
|------|-------------|
| `dascim_all.yaml` | Task config to run ALL 66 configurations combined |
| `utils.py` | `custom_dataset()` function that loads all configs and fixes answer indexing |
| `configs.py` | Script to generate YAML files (development only, not used at runtime) |

### default/ Folder

| File | Description |
|------|-------------|
| `_default_template_yaml` | Base template with common settings (metrics, output_type, etc.) |
| `_dascim_*.yaml` | Group definitions that aggregate subtasks by category |
| `base_dascim_*choice.yaml` | Base configs for 2/3/4 choice questions |
| `dascim_*.yaml` | Individual subject+choice task definitions |
| `utils.py` | `custom_dataset(dataset_name)` for loading individual configs |

## 🔧 Configuration Details

### Answer Indexing Fix
The dataset has some questions with 1-indexed answers (1,2,3,4) instead of 0-indexed (0,1,2,3). 
The `custom_dataset()` functions automatically fix this.

### Dataset Source
- HuggingFace: `mkonomi/greek_mmlu_configs`
- 66 configurations (subject + choice count combinations)
- ~15,000 total questions

## 📈 Example Results

```
|Groups          |Metric|Value |
|----------------|------|------|
|dascim_stem     |acc   |0.4907|
|dascim_humanities|acc  |0.5234|
|dascim_all      |acc   |0.4800|
```

## 🛠️ Regenerating YAMLs

If you need to regenerate task YAMLs (not usually necessary):

```bash
python configs.py \
    --base_yaml_path _default_template_yaml \
    --process_data \
    --input_data_dir /path/to/json/data \
    --output_data_dir ./split_data
```

