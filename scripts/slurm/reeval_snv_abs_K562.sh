#!/bin/bash
# Re-evaluate existing K562 chr-split checkpoints with the updated
# test_episomal_mpra.py (which now also computes snv_abs_alt). Populates
# the 4th-column bar (SNV / Alt Allele Pearson r) without retraining
# anything.
#
# 9 array tasks: 3 AG fine-tuned seeds + 3 Enformer probing seeds + 3
# Enformer fine-tuned seeds. (Malinois already computes snv_abs inline.)
# Wall ≈ 1 task (~20-30 min) since they're independent.

set -euo pipefail
REPO=/grid/wsbs/home_norepl/christen/alphagenome_FT_MPRA
ALB_REPO=/grid/wsbs/home_norepl/christen/ALBench-S2F
cd "$REPO"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

JOBFILE=$(mktemp)
cat > "$JOBFILE" <<'EOF'
#!/bin/bash
#SBATCH --job-name=reeval_snv_abs_K562
#SBATCH --output=__REPO__/logs/%x-%A_%a.out
#SBATCH --error=__REPO__/logs/%x-%A_%a.err
#SBATCH --partition=gpuq
#SBATCH --qos=slow_nice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-8

cd __REPO__
set +u; source /etc/profile.d/modules.sh; set -u
module load EB5 2>/dev/null || true
# Reuse the ALBench-S2F venv (has all deps + AG)
source __ALB_REPO__/.venv/bin/activate
export PYTHONPATH="$PWD:__ALB_REPO__${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Array layout: 3 model types × 3 seeds = 9 tasks
MODEL_TYPES=(ag_finetuned ag_finetuned ag_finetuned \
             enformer_probing enformer_probing enformer_probing \
             enformer_finetuned enformer_finetuned enformer_finetuned)
SEEDS=(42 1042 2042 42 1042 2042 42 1042 2042)

MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CELL=K562

# Resolve checkpoint path per model_type
case "$MODEL_TYPE" in
    ag_finetuned)
        CKPT="results/models/checkpoints/episomal/${CELL}/episomal-${CELL}-seed${SEED}"
        ;;
    enformer_probing)
        # stage1 best ckpt (auto-select via find)
        CKPT_DIR="results/models/checkpoints/episomal_enformer_production_v2/${CELL}/enformer-prod2-${CELL}-seed${SEED}/stage1"
        CKPT=$(ls -1 "$CKPT_DIR"/best-stage1-*.ckpt 2>/dev/null | head -1)
        ;;
    enformer_finetuned)
        CKPT_DIR="results/models/checkpoints/episomal_enformer_production_v2/${CELL}/enformer-prod2-${CELL}-seed${SEED}/stage2"
        CKPT=$(ls -1 "$CKPT_DIR"/best-stage2-*.ckpt 2>/dev/null | head -1)
        ;;
esac

if [ -z "${CKPT:-}" ] || [ ! -e "$CKPT" ]; then
    echo "[task $SLURM_ARRAY_TASK_ID] WARN: missing ckpt for $MODEL_TYPE seed=$SEED, skipping"
    exit 0
fi

OUT_DIR="results/episomal_predictions_v3_snv_abs"
mkdir -p "$OUT_DIR"

echo "[task $SLURM_ARRAY_TASK_ID] $MODEL_TYPE seed=$SEED ckpt=$CKPT"

# Use ALBench venv's python directly (uv run defaults to this repo's
# pyproject.toml which doesn't have the AG/Enformer deps).
python -u scripts/test_episomal_mpra.py \
    --model_type $MODEL_TYPE \
    --checkpoint_path "$CKPT" \
    --cell_type $CELL \
    --data_path __ALB_REPO__/data/k562 \
    --output_dir $OUT_DIR \
    --run_name "${MODEL_TYPE}-${CELL}-seed${SEED}"
EOF

sed -i.bak "s|__REPO__|$REPO|g; s|__ALB_REPO__|$ALB_REPO|g" "$JOBFILE"
rm -f "$JOBFILE.bak"

JID=$($SBATCH --parsable "$JOBFILE")
rm -f "$JOBFILE"
echo "K562 snv_abs re-eval array → $JID (9 parallel tasks on slow_nice)"
echo "Outputs land in $REPO/results/episomal_predictions_v3_snv_abs/"
