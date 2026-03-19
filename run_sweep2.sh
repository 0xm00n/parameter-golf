#!/bin/bash
cd ~/projects/parameter-golf
source .venv/bin/activate

run_config() {
    local name=$1 layers=$2 dim=$3 heads=$4 kvheads=$5
    echo "=== $name (layers=$layers dim=$dim) ==="
    RUN_ID=$name NUM_LAYERS=$layers MODEL_DIM=$dim NUM_HEADS=$heads NUM_KV_HEADS=$kvheads \
        ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
        python3 train_gpt.py 2>&1 | grep -E "model_params|final_int8_zlib_roundtrip_exact|peak memory|Serialized model int8"
}

run_config max_4L_960W 4 960 8 4 &
run_config try_3L_1024W 3 1024 8 4 &
wait
run_config mid_4L_896W 4 896 8 4 &
wait
echo "=== ALL DONE ==="
