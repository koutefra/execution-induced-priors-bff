#!/bin/bash
set -e

START=0
END=99

for SEED in $(seq $START $END); do
    GPU=$(( SEED % 4 ))
    echo "[reinit] seed $SEED → GPU $GPU"
    ./run_seed.sh reinit "$SEED" "$GPU" &
    
    # Limit concurrency to 4 jobs (1 per GPU)
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n
    fi
done

wait
echo "All reinit runs finished."
