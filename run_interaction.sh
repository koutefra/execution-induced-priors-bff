#!/bin/bash
set -e

START=0
END=19

for SEED in $(seq $START $END); do
    GPU=$(( SEED % 4 ))
    echo "[interaction] seed $SEED → GPU $GPU"
    ./run_seed.sh interaction "$SEED" "$GPU" &
    
    # Limit concurrency to 4 jobs (1 per GPU)
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n
    fi
done

wait
echo "All interaction runs finished."
