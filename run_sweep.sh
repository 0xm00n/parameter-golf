#!/bin/bash
cd ~/projects/parameter-golf
source .venv/bin/activate

run_config() {
    local name=$1 layers=$2 dim=$3 heads=$4 kvheads=$5
    echo "=== $name (layers=$layers dim=$dim) ==="
    RUN_ID=$name NUM_LAYERS=$layers MODEL_DIM=$dim NUM_HEADS=$heads NUM_KV_HEADS=$kvheads \
        ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
        python3 train_gpt.py 2>&1 | grep -E "model_params|final_int8_zlib_roundtrip_exact|peak memory"
}

run_config deep_narrow 14 384 8 4 &
run_config shallow_wide 6 640 8 4 &
wait
run_config very_deep 18 320 8 4 &
run_config very_wide 4 832 8 4 &
wait
echo "=== ALL DONE ==="
