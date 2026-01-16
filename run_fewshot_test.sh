#!/bin/bash
# Test different few-shot settings using dascim_all (no sample limitations)

MODEL="ilsp/Meltemi-7B-v1"
CACHE_DIR="/home/mersin-konomi/cache"
OUTPUT_PATH="/home/mersin-konomi/eval_results"
BATCH_SIZE=1

# Few-shot values to test
FEWSHOTS=(0 3 5 10 25)

echo "============================================"
echo "DASCIM Few-shot Comparison Test"
echo "Model: $MODEL"
echo "Task: dascim_all (~15,000 questions)"
echo "Few-shots to test: ${FEWSHOTS[*]}"
echo "============================================"

for fewshot in "${FEWSHOTS[@]}"; do
    echo ""
    echo "############################################"
    echo "Testing ${fewshot}-shot"
    echo "############################################"
    
    python3 -m lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
        --tasks dascim_all \
        --batch_size $BATCH_SIZE \
        --trust_remote_code \
        --num_fewshot $fewshot \
        --output_path "$OUTPUT_PATH" \
        --device cuda:0 \
        --log_samples
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: ${fewshot}-shot"
    else
        echo "✗ Failed: ${fewshot}-shot"
    fi
done

echo ""
echo "============================================"
echo "ALL FEW-SHOT TESTS COMPLETED!"
echo "Results saved to: $OUTPUT_PATH"
echo "============================================"

