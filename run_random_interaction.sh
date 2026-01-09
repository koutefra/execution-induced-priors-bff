#!/bin/bash
set -e

START=0
END=99
MAX_JOBS=4

for SEED in $(seq $START $END); do
    echo "[random interaction] launching seed $SEED"
    ./run_seed.sh random "$SEED" &
    
    if (( $(jobs -r | wc -l) >= MAX_JOBS )); then
        wait -n
    fi
done

wait
echo "All random interaction runs finished."
