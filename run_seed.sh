#!/bin/bash
set -e

MODE=$1       # "reinit", "interaction", or "random"
SEED=$2       # integer

# Valid compute GPUs on your DGX (skip GPU 3 – it's display)
GPUS=(0 1 2 4)

LOCKDIR="/tmp/gpu_locks"
mkdir -p "$LOCKDIR"

# ---- acquire a free GPU ----
while true; do
  for GPU in "${GPUS[@]}"; do
    LOCK="$LOCKDIR/gpu_${GPU}.lock"
    if ( set -o noclobber; echo "$$" > "$LOCK" ) 2>/dev/null; then
      GPU_ID=$GPU
      export CUDA_VISIBLE_DEVICES=$GPU_ID
      trap 'status=$?; rm -f "$LOCK"; exit $status' INT TERM EXIT
      break 2
    fi
  done
  sleep 1
done
# ---- GPU acquired ----

OUTDIR="runs/${MODE}"
mkdir -p "$OUTDIR"

STDOUT="${OUTDIR}/out_${SEED}.log"
LOGFILE="${OUTDIR}/log_${SEED}.log"
STDERR="${OUTDIR}/err_${SEED}.log"

echo "[${MODE}] seed ${SEED} → GPU ${GPU_ID}"

if [ "$MODE" = "reinit" ]; then
    bin/main --lang bff_noheads --num 131072 \
        --reinit_each_epoch --eval_selfrep --print_selfrep \
        --print_interval 1 --log_interval 1 --max_epochs 16000 \
        --disable_output --seed "${SEED}" \
        --log "$LOGFILE" \
        > "$STDOUT" 2> "$STDERR"

elif [ "$MODE" = "interaction" ]; then
    bin/main --lang bff_noheads --num 131072 \
        --eval_selfrep --print_selfrep \
        --print_interval 512 --log_interval 1 --max_epochs 16000 \
        --mutation_prob 0.00024 --seed "${SEED}" \
        --log "$LOGFILE" \
        > "$STDOUT" 2> "$STDERR"

elif [ "$MODE" = "random" ]; then
    bin/main --lang bff_noheads --num 131072 \
        --eval_selfrep --print_selfrep \
        --print_interval 512 --log_interval 1 --max_epochs 16000 \
        --mutation_prob 0.00024 --random_partner_interaction \
        --seed "${SEED}" \
        --log "$LOGFILE" \
        > "$STDOUT" 2> "$STDERR"
else
    echo "Unknown mode: $MODE"
    exit 1
fi
