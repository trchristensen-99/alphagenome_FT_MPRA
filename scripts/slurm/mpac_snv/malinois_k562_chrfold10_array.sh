#!/bin/bash
# Submit 10 parallel Malinois K562 chr-fold training jobs as a SLURM array.
# Each fold uses a different val_chr (all excluding chr 7+13 = test).
# 10 parallel jobs on slow_nice → wall time ≈ 1 fold's training time
# (~1.5h) instead of ~15h serialized.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

# fold N uses val_chr from this list at index N
VAL_CHRS=(1 2 3 4 5 6 8 9 10 11)

OUT_BASE=results/snv_eval/malinois_k562_chrfold10
mkdir -p "$OUT_BASE" "$REPO/logs"

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<'EOF'
#!/bin/bash
#SBATCH --job-name=malinois_k562_chrfold
#SBATCH --output=__REPO__/logs/%x-%A_%a.out
#SBATCH --error=__REPO__/logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=0-9

cd __REPO__
source .venv/bin/activate
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

VAL_CHRS=(1 2 3 4 5 6 8 9 10 11)
VAL_CHR=${VAL_CHRS[$SLURM_ARRAY_TASK_ID]}
SEED=$((42 + SLURM_ARRAY_TASK_ID))
OUT_DIR=__OUT_BASE__/fold${SLURM_ARRAY_TASK_ID}_val_chr${VAL_CHR}
mkdir -p "$OUT_DIR"

echo "[malinois fold $SLURM_ARRAY_TASK_ID] val_chr=$VAL_CHR  seed=$SEED  out=$OUT_DIR"

python experiments/train_malinois_k562.py \
    ++cell_line=k562 \
    ++chr_split=true \
    ++val_chrs=$VAL_CHR \
    ++test_chrs=7,13 \
    ++seed=$SEED \
    ++output_dir=$OUT_DIR
EOF

# substitute placeholders (avoid heredoc expansion issues)
sed -i.bak "s|__REPO__|$REPO|g; s|__OUT_BASE__|$OUT_BASE|g" "$JOBFILE"
rm -f "$JOBFILE.bak"

JID=$($SBATCH --parsable "$JOBFILE")
rm -f "$JOBFILE"
echo "Malinois K562 chr-fold 10-array → $JID (10 parallel folds)"
echo "  Each fold uses val_chr ∈ {1,2,3,4,5,6,8,9,10,11}, test=chr7+13"
