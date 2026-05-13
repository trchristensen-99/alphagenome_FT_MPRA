# MPAC-style SNV evaluation

Self-contained scripts for the SNV/skew bar-plot columns. Adapted from
ALBench-S2F (`scripts/preflight/snv_eval/`).

## What this does (paper-style protocol)

Following Butts et al. (bioRxiv 2025.04.16) **MPAC** = *Malinois with Parallel
Aggregated Cross-validation*. Two ingredients:

1. **Chromosome-fold ensemble.** Train N models that each hold out a different
   *val_chr* (one chromosome). All folds also hold out *test_chrs* (the eval
   chromosomes — chr 7+13 here). For the bar plot we use **N = 10** folds
   rather than MPAC's full 11×10 grid because we only need cross-fold safety
   on the eval chromosomes, not genome-wide.

2. **Sliding-window + RC averaging at inference.** For each (ref, alt) variant:
   - slide the variant within an N-window crop of the model's input,
   - predict both forward and reverse-complement strands,
   - average across (N_windows × 2 strands × N_models),
   - skew = mean_alt − mean_ref.

The mean reduces three independent sources of noise: model-init, positional
sensitivity, strand orientation.

## Reproducing the bar-plot columns

```
# 1. Train the 10-fold ensemble (one SLURM job, ~1.5-6h depending on arch):
sbatch scripts/slurm/mpac_snv/malinois_k562_chrfold10_array.sh     # 10 parallel folds
sbatch scripts/slurm/mpac_snv/dreamrnn_k562_chrfold10_array.sh

# 2. Inference: 4 bar-plot metrics on chr 7+13 SNV pairs.
python scripts/mpac_snv/predict_skew.py \
    --snv_parquet outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/snv_pairs.parquet \
    --chrs 7,13 \
    --models results/snv_eval/malinois_k562_chrfold10/fold*/best.pt \
    --arch malinois \
    --output_dir results/snv_eval/k562_chr7_13_malinois_10fold
```

The output `summary.json` reports four Pearson r's:
- `ref_pearson`   — reference activity (predicted vs ref_log2FC) → "Genomic Reference" column
- `alt_pearson`   — alt-allele absolute activity (predicted vs alt_log2FC) → "SNV (Alt Allele)" column
- `skew_pearson`  — predicted skew vs empirical Log2Skew → "SNV Effects" column
- (designed-test set is evaluated separately by `test_episomal_mpra.py`)

## Why N=10 instead of MPAC's N=110

MPAC needs 110 models because any genome position must have 10 models that did
not see its source chromosome. For our held-out chr 7+13 evaluation, **all 10
folds already exclude chr 7+13** — the only thing the per-fold val_chr varies
is which extra chromosome got held out. 10-model ensemble averaging gives the
~√10 noise reduction; the missing 11× redundancy is only useful if you want to
predict variants on chr 1-22 (which we don't).

## Labels

`--label_source real` uses raw `K562_log2FC` / `HepG2_log2FC` / `SKNSH_log2FC`
labels — required for bar-plot model comparisons (apples-to-apples vs Malinois
paper). `--label_source ag_oracle` uses denoised AG pseudolabels, only for
scaling-law / HP-search work.

**WARNING — hashFrag oracles**: AG models trained on hashFrag splits (any
sequence-similarity-based split) saw chr 7+13 sequences during training and
**must not** be used for the SNV/skew bars. Use chr-split models only.
