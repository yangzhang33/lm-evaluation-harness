#!/bin/bash
MODEL="ilsp/Meltemi-7B-v1"
CACHE_DIR="/home/mersin-konomi/cache"
OUTPUT_PATH="/home/mersin-konomi/eval_results"
BATCH_SIZE=1

echo "============================================"
echo "Starting DASCIM Evaluation"
echo "Model: $MODEL"
echo "============================================"


echo ""
echo "############################################"
echo "FEW-SHOT: 0"
echo "############################################"

echo ""
echo "--------------------------------------------"
echo "Running: dascim_stem (0-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_stem" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_stem (0-shot)"
else
    echo "Failed: dascim_stem (0-shot)"
fi

echo ""
echo "--------------------------------------------"
echo "Running: dascim_humanities (0-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_humanities" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_humanities (0-shot)"
else
    echo "Failed: dascim_humanities (0-shot)"
fi

echo ""
echo "--------------------------------------------"
echo "Running: dascim_social_sciences (0-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_social_sciences" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_social_sciences (0-shot)"
else
    echo "Failed: dascim_social_sciences (0-shot)"
fi

echo ""
echo "--------------------------------------------"
echo "Running: dascim_other (0-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_other" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 0 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_other (0-shot)"
else
    echo "Failed: dascim_other (0-shot)"
fi

echo ""
echo "############################################"
echo "FEW-SHOT: 3"
echo "############################################"

echo ""
echo "--------------------------------------------"
echo "Running: dascim_stem (3-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_stem" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 3 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_stem (3-shot)"
else
    echo "Failed: dascim_stem (3-shot)"
fi

echo ""
echo "--------------------------------------------"
echo "Running: dascim_humanities (3-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_humanities" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 3 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_humanities (3-shot)"
else
    echo "Failed: dascim_humanities (3-shot)"
fi

echo ""
echo "--------------------------------------------"
echo "Running: dascim_social_sciences (3-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_social_sciences" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 3 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_social_sciences (3-shot)"
else
    echo "Failed: dascim_social_sciences (3-shot)"
fi

echo ""
echo "--------------------------------------------"
echo "Running: dascim_other (3-shot)"
echo "--------------------------------------------"

python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=$MODEL,cache_dir=$CACHE_DIR" \
    --tasks "dascim_other" \
    --batch_size $BATCH_SIZE \
    --trust_remote_code \
    --num_fewshot 3 \
    --output_path "$OUTPUT_PATH" \
    --device cuda:0 \
    --log_samples

if [ $? -eq 0 ]; then
    echo "Completed: dascim_other (3-shot)"
else
    echo "Failed: dascim_other (3-shot)"
fi

echo ""
echo "============================================"
echo "ALL EVALUATIONS COMPLETED!"
echo "Results saved to: $OUTPUT_PATH"
echo "============================================"
