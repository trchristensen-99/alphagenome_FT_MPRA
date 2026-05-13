#!/bin/bash
# Submit 10 parallel DREAM-RNN K562 chr-fold training jobs as a SLURM array.
# Each fold = its own SLURM job → no inter-fold CUDA contention (the
# crashes at k=2/k=4 with shared GPU were because LSTM memory growth is
# unpredictable when 2+ models train concurrently on one GPU).
# Wall time ≈ 1 fold (~2-3h) instead of ~15h serialized.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

OUT_BASE=results/snv_eval/dream_rnn_k562_chrfold10
mkdir -p "$OUT_BASE" "$REPO/logs"

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<'EOF'
#!/bin/bash
#SBATCH --job-name=dreamrnn_k562_chrfold
#SBATCH --output=__REPO__/logs/%x-%A_%a.out
#SBATCH --error=__REPO__/logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=0-9

cd __REPO__
source .venv/bin/activate
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export HP_FAST=1

VAL_CHRS=(1 2 3 4 5 6 8 9 10 11)
VAL_CHR=${VAL_CHRS[$SLURM_ARRAY_TASK_ID]}
SEED=$((42 + SLURM_ARRAY_TASK_ID))
OUT_DIR=__OUT_BASE__/fold${SLURM_ARRAY_TASK_ID}_val_chr${VAL_CHR}
mkdir -p "$OUT_DIR"

# Skip if already complete from a prior (failed) run
if [ -f "$OUT_DIR/result.json" ]; then
    echo "[skip fold $SLURM_ARRAY_TASK_ID val_chr=$VAL_CHR] result.json already exists"
    exit 0
fi

echo "[dream_rnn fold $SLURM_ARRAY_TASK_ID] val_chr=$VAL_CHR  seed=$SEED  out=$OUT_DIR"

uv run --no-sync python -u scripts/preflight/run_single.py \
    --arch dream_rnn --d_train 0 --seed $SEED \
    --epochs 60 --early_stop_patience 10 \
    --augmentations rev_complement \
    --label_source real \
    --val_chrs $VAL_CHR --test_chrs 7,13 \
    --output_dir $OUT_DIR \
    --fast \
    --hp lr=0.001 batch_size=512 weight_decay=0.01 \
         hidden_dim=320 cnn_filters=160 num_lstm_layers=1 \
         dropout_cnn=0.2 dropout_lstm=0.3
EOF

sed -i.bak "s|__REPO__|$REPO|g; s|__OUT_BASE__|$OUT_BASE|g" "$JOBFILE"
rm -f "$JOBFILE.bak"

JID=$($SBATCH --parsable "$JOBFILE")
rm -f "$JOBFILE"
echo "DREAM-RNN K562 chr-fold 10-array → $JID (10 parallel folds on slow_nice)"
