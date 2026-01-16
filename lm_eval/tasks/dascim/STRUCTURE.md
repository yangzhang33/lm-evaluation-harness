# DASCIM Task Structure Overview

## 📊 Hierarchy Diagram

```
dascim/
│
├── ROOT LEVEL (Simple, Unified Approach)
│   ├── dascim_all.yaml
│   │   └── Single task: ALL ~15k questions combined
│   │       └── Uses: utils.py (loads all configs at once)
│   │
│   └── utils.py
│       └── custom_dataset() - loads ALL 66 configs, concatenates them
│
└── default/ (Granular, Grouped Approach)
    │
    ├── BASE TEMPLATES (Shared Configuration)
    │   ├── _default_template_yaml
    │   │   └── Common settings: metrics, output_type, doc_to_* functions
    │   │
    │   ├── base_dascim_2choice.yaml
    │   ├── base_dascim_3choice.yaml
    │   └── base_dascim_4choice.yaml
    │       └── Includes: _default_template_yaml
    │
    ├── INDIVIDUAL TASKS (66 files)
    │   ├── dascim_Medicine_4choice.yaml
    │   │   ├── Includes: base_dascim_4choice.yaml
    │   │   ├── tag: "dascim_stem_tasks"  ← Groups into STEM category
    │   │   ├── task: "dascim_Medicine_4choice"
    │   │   └── dataset_name: "Medicine_4choice"
    │   │
    │   ├── dascim_Physics_2choice.yaml
    │   ├── dascim_Biology_4choice.yaml
    │   ├── dascim_Greek_History_4choice.yaml
    │   └── ... (63 more)
    │
    ├── CATEGORY GROUPS (4 groups)
    │   ├── _dascim_stem.yaml
    │   │   ├── group: "dascim_stem"
    │   │   ├── task: ["dascim_stem_tasks"]  ← References TAG, not individual tasks
    │   │   └── aggregate_metric_list: weighted average
    │   │
    │   ├── _dascim_humanities.yaml
    │   ├── _dascim_social_sciences.yaml
    │   └── _dascim_other.yaml
    │
    ├── TOP-LEVEL GROUP
    │   └── _dascim.yaml
    │       ├── group: "dascim"
    │       ├── task: [
    │       │     "dascim_humanities",
    │       │     "dascim_other",
    │       │     "dascim_social_sciences",
    │       │     "dascim_stem"
    │       │   ]  ← References GROUPS, not tags
    │       └── aggregate_metric_list: weighted average
    │
    └── utils.py
        └── custom_dataset(dataset_name) - loads specific config + cross-subject few-shot
```

## 🔑 Key Concepts

### 1. **Two Approaches**

#### **A. Root Level: `dascim_all`**
- **Purpose**: Quick, unified evaluation
- **Structure**: Single YAML file
- **Data Loading**: Loads ALL 66 configurations at once
- **Output**: One accuracy metric for everything
- **Use Case**: `--tasks dascim_all`

#### **B. Default Folder: Granular Groups**
- **Purpose**: Detailed, category-specific evaluation
- **Structure**: Hierarchical (Individual → Category → Top-level)
- **Data Loading**: Subject-specific test sets, cross-subject few-shot
- **Output**: Per-subject, per-category, and aggregated metrics
- **Use Case**: `--tasks dascim` or `--tasks dascim_stem`

### 2. **Configuration Inheritance**

```
_default_template_yaml (base settings)
    ↓
base_dascim_4choice.yaml (includes template)
    ↓
dascim_Medicine_4choice.yaml (includes base, adds subject-specific config)
```

### 3. **Tagging System**

Individual tasks use `tag` to mark their category:
- `tag: "dascim_stem_tasks"` → All STEM subjects (Medicine, Physics, Biology, etc.)
- `tag: "dascim_humanities_tasks"` → All Humanities subjects
- `tag: "dascim_social_sciences_tasks"` → All Social Sciences
- `tag: "dascim_other_tasks"` → Other subjects

### 4. **Group vs Tag**

- **Tag**: A label that tasks can have. When you reference a tag in a group's `task` list, it includes all tasks with that tag.
- **Group**: A named collection that aggregates results. Can reference:
  - Individual task names
  - Tag names (includes all tasks with that tag)
  - Other group names (nested groups)

### 5. **Aggregation Levels**

```
Level 1: Individual Tasks
  └── dascim_Medicine_4choice → accuracy: 0.65

Level 2: Category Groups
  └── dascim_stem → weighted average of all STEM tasks

Level 3: Top-Level Group
  └── dascim → weighted average of all 4 categories
```

## 📝 File Types Explained

### Root Level Files

| File | Purpose |
|------|---------|
| `dascim_all.yaml` | Single task config for all questions |
| `utils.py` | Loads all configs, concatenates, fixes indexing |

### Default Folder Files

| File Pattern | Purpose | Example |
|--------------|---------|---------|
| `_default_template_yaml` | Base template with common settings | Metrics, output_type |
| `base_dascim_*choice.yaml` | Choice-count specific base configs | 2/3/4 choice formatting |
| `dascim_*_*choice.yaml` | Individual subject+choice tasks | Medicine_4choice, Physics_2choice |
| `_dascim_*.yaml` | Category group configs | STEM, Humanities, etc. |
| `_dascim.yaml` | Top-level group (all categories) | Aggregates all 4 categories |
| `utils.py` | Subject-specific loading + cross-subject few-shot | |

## 🎯 Usage Examples

### Run Everything at Once
```bash
--tasks dascim_all
# Result: Single accuracy metric
```

### Run by Category
```bash
--tasks dascim_stem
# Result: Weighted average of all STEM subjects
```

### Run Top-Level Group
```bash
--tasks dascim
# Result: Weighted average of all 4 categories
```

### Run Individual Subject
```bash
--tasks dascim_Medicine_4choice
# Result: Accuracy for Medicine 4-choice questions only
```

## 🔄 Data Flow

### `dascim_all` Flow:
```
utils.custom_dataset()
  → Loads all 66 configs from HF
  → Concatenates into one dataset
  → Returns DatasetDict with 'test' split
  → Single evaluation run
```

### `dascim` (grouped) Flow:
```
_dascim.yaml (top group)
  → References 4 category groups
    → Each category references a tag
      → Tag includes all matching individual tasks
        → Each task uses utils.custom_dataset(dataset_name)
          → Loads specific config for test
          → Loads ALL configs for few-shot
          → Returns DatasetDict with 'test' and 'fewshot' splits
  → Aggregates results with weights
```

## 💡 Key Differences Summary

| Aspect | `dascim_all` | `dascim` (group) |
|--------|--------------|------------------|
| **Structure** | Single task | Hierarchical groups |
| **Data Loading** | All at once | Subject-specific |
| **Few-shot** | Same dataset | Cross-subject pool |
| **Results** | One metric | Per-subject + aggregated |
| **Flexibility** | Simple | Granular analysis |
| **Use Case** | Quick overall score | Detailed breakdown |

