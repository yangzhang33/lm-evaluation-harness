#!/bin/bash
# Configuration
CACHE_DIR="/home/mersin-konomi/cache"
BASE_OUTPUT_PATH="/home/mersin-konomi/eval_results"
WORK_DIR="/home/mersin-konomi/model_eval/lm-evaluation-harness"

# Create output directories
mkdir -p "$CACHE_DIR"
mkdir -p "$BASE_OUTPUT_PATH/krikri_cot"

# Model and Task
MODEL_NAME="ilsp/Llama-Krikri-8B-Instruct"
TASK="greekmmlu_cot"
FEW_SHOT=0
OUTPUT_PATH="$BASE_OUTPUT_PATH/krikri_cot"

echo "Starting evaluation for $MODEL_NAME on $TASK"
echo "Cache directory: $CACHE_DIR"
echo "Output path: $OUTPUT_PATH"
echo "================================================"

# Build and run the command
cd "$WORK_DIR"
python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL_NAME,device_map=auto" \
    --tasks "$TASK" \
    --batch_size 8 \
    --trust_remote_code \
    --num_fewshot "$FEW_SHOT" \
    --output_path "$OUTPUT_PATH" \
    --log_samples \
    "$@"

if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed!"
    exit 1
fi
