#!/bin/bash
# Configuration
CACHE_DIR="/home/mersin-konomi/cache"
BASE_OUTPUT_PATH="/home/mersin-konomi/eval_results"
WORK_DIR="/home/mersin-konomi/model_eval/lm-evaluation-harness"

# Create output directories
mkdir -p "$CACHE_DIR"
mkdir -p "$BASE_OUTPUT_PATH/llama_8b_split"

# Model and Task
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
TASK="greekmmlu_split"
FEW_SHOT=5
OUTPUT_PATH="$BASE_OUTPUT_PATH/llama_8b_split"

echo "Starting evaluation for $MODEL_NAME on $TASK"
echo "Using device_map=auto (Single Process / Model Parallelism)"
echo "Output path: $OUTPUT_PATH"
echo "================================================"

# Build and run the command
cd "$WORK_DIR"

# Using python3 directly with device_map=auto to avoid data-parallel crashes on tiny tasks
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_NAME,dtype=bfloat16,device_map=auto" \
    --tasks "$TASK" \
    --batch_size auto \
    --trust_remote_code \
    --num_fewshot "$FEW_SHOT" \
    --output_path "$OUTPUT_PATH" \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed!"
    exit 1
fi
