#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Define the log file
LOG_FILE="logs/run_${TIMESTAMP}.log"

# Redirect all output to the log file and also display on console
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting script at $(date)"

# Exit the script if any command fails
set -e

# Your existing script content here, updated based on launch.json
DATA_ROOT=""
OUTPUT_ROOT="./results_gsm8k"
MODEL="gpt"
API_KEY=""
API_URL="https://api.openai.com/v1/chat/completions"
TEST_SPLIT="test"
SHOT_NUMBER=0
SEED=10
METHOD="ours"
TEST_NUMBER=100
NUM_WORKERS=1
TEMPERATURE=0.5
TOP_P=0.4
MAX_TOKENS=1000
PROMPT_FORMAT="CQM-A"

# ToT
TASK="gsm8k"
METHOD_GENERATE="sample"
METHOD_EVALUATE="vote"
METHOD_SELECT="greedy"
N_GENERATE_SAMPLE="5"
N_EVALUATE_SAMPLE="5"
N_SELECT_SAMPLE="1"
PROMPT_SAMPLE="cot"

# Run the Python script
python ./run_gsm8k_multiple_verifiers_base.py \
    --data_root "$DATA_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model "$MODEL" \
    --api_key "$API_KEY" \
    --api_url "$API_URL" \
    --test_split "$TEST_SPLIT" \
    --shot_number "$SHOT_NUMBER" \
    --seed "$SEED" \
    --method "$METHOD" \
    --test_number "$TEST_NUMBER" \
    --num_workers "$NUM_WORKERS" \
    --prompt_format "$PROMPT_FORMAT" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --task "$TASK" \
    --method_generate "$METHOD_GENERATE" \
    --method_evaluate "$METHOD_EVALUATE" \
    --method_select "$METHOD_SELECT" \
    --n_generate_sample "$N_GENERATE_SAMPLE" \
    --n_evaluate_sample "$N_EVALUATE_SAMPLE" \
    --n_select_sample "$N_SELECT_SAMPLE" \
    --prompt_sample "$PROMPT_SAMPLE"

echo "Script completed at $(date)"
