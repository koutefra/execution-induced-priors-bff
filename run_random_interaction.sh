#!/bin/bash
set -e

START=0
END=99

GPUS=(0 1 2 4)

for SEED in $(seq $START $END); do
    GPU=${GPUS[$(( SEED % ${#GPUS[@]} ))]}
    echo "[random interaction] seed $SEED → GPU $GPU"
    ./run_seed.sh random "$SEED" "$GPU" &
    
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n
    fi
done

wait
echo "All random interaction runs finished."
