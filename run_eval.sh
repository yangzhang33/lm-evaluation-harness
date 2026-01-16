#!/bin/bash
# chmod +x /home/mersin-konomi/model_eval/lm-evaluation-harness/run_eval.sh

# Configuration
CACHE_DIR="/home/mersin-konomi/cache"
BASE_OUTPUT_PATH="/home/mersin-konomi/eval_results"
WORK_DIR="/home/mersin-konomi/model_eval/lm-evaluation-harness"

# Create output directories
mkdir -p "$CACHE_DIR"
mkdir -p "$BASE_OUTPUT_PATH/results_0_shot"
mkdir -p "$BASE_OUTPUT_PATH/results_5_shot"

# Define models to test (Meta Llama models)
declare -a MODELS=(
    "THUDM/glm-4-9b"
    "THUDM/glm-4-9b-chat"
    "THUDM/glm-4-9b-chat-1m"
    "THUDM/chatglm3-6b"
)
# Few-shot settings
declare -a FEW_SHOTS=(5)

# Task name
TASK="greekmmlu_cot"

echo "Starting evaluation for ${#MODELS[@]} model(s)"
echo "Cache directory: $CACHE_DIR"
echo "Output paths: $BASE_OUTPUT_PATH/results_0_shot"
echo "================================================"

# Run evaluation for each model
for MODEL_NAME in "${MODELS[@]}"; do
    # Extract short model name for folder
    MODEL_SHORT=$(basename "$MODEL_NAME")
    
    echo "Starting evaluation for model: $MODEL_NAME ($MODEL_SHORT)"
    echo "================================================"
    

    declare -a TASKS=("greekmmlu_qa")
    # declare -a TASKS=("greekmmlu_cot_greek_literature") # Targeted verification
    
    for TASK in "${TASKS[@]}"; do
        echo "Starting task: $TASK"
        
        # Run evaluation for each few-shot setting
        for FEW_SHOT in "${FEW_SHOTS[@]}"; do
            # Set output path based on few-shot setting and task
            OUTPUT_PATH="$BASE_OUTPUT_PATH/results_${FEW_SHOT}_shot/$MODEL_SHORT/$TASK"
            mkdir -p "$OUTPUT_PATH"
            
            echo "Running $TASK with $FEW_SHOT few-shot examples..."
            echo "Output: $OUTPUT_PATH"
            
            # Build and run the command
            cd "$WORK_DIR"
            /home/mersin-konomi/miniconda/bin/python3 -m lm_eval \
                --model hf \
                --model_args "pretrained=$MODEL_NAME,device_map=auto,max_length=128000" \
                --tasks "$TASK" \
                --batch_size auto \
                --trust_remote_code \
                --num_fewshot "$FEW_SHOT" \
                --output_path "$OUTPUT_PATH" \
                --log_samples
            
            if [ $? -ne 0 ]; then
                echo "Error: Evaluation failed for $TASK ($FEW_SHOT-shot) on model $MODEL_NAME"
                # Continue to next task/setting
                continue 
            fi
            
            echo "Completed $TASK ($FEW_SHOT-shot) for $MODEL_NAME"
            echo "--------------------------------"
        done
    done
    
    echo "Completed all evaluations for model: $MODEL_NAME"
    echo "================================================"
done

echo "All evaluations completed successfully for all models!"
echo ""
echo "Results structure:"
echo "$BASE_OUTPUT_PATH/"
echo "├── results_0_shot/"
echo "│   ├── Llama-3.2-3B-Instruct/"
echo "│   ├── Llama-3.1-8B-Instruct/"
echo "│   └── Llama-3.1-70B-Instruct/"
echo "└── results_5_shot/"
echo "    ├── Llama-3.2-3B-Instruct/"
echo "    └── ..."