python run.py \
    --task gsm8k \
    --task_start_index 0 \
    --task_end_index 500 \
    --n_generate_sample 1 \
    --prompt_sample standard \
    --temperature 1.0 \
    --naive_run  # Just specify the flag, no value needed
    ${@}
