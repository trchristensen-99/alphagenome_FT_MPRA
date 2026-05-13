"""Train an N-fold chromosome-split ensemble for one (arch, cell type).

Each fold differs only in which chromosome is held as VALIDATION. All folds
exclude chr 7 + 13 from training (those stay as the held-out SNV test set).
This is the MPAC-style ensemble adapted for our chr-7+13 SNV eval:

  fold 0: train = autosomes minus {7, 13, 1}, val = chr 1
  fold 1: train = autosomes minus {7, 13, 2}, val = chr 2
  ...
  fold 9: train = autosomes minus {7, 13, 10}, val = chr 10

For SMALL archs (LegNet, DREAM-RNN, foundation-model probing heads): builds
a single configs.json + runs the in-process N-model trainer for the entire
ensemble in one Python process on one GPU.

For LARGE archs (Enformer/AG fine-tune): submit one SLURM job per fold.

Inputs:
  --pool_dir: oracle_pseudolabels_*/pool/ — contains train.parquet / val.parquet
              with a `locus` column we can split on.
  Or use existing chromosome-keyed split logic in run_single.py if pool not
  pre-split.

Usage:
  # Small arch (in-process)
  python -m scripts.preflight.snv_eval.train_chrfold_ensemble \\
    --arch legnet --cell k562 --n_folds 10 \\
    --hp_overrides lr=3e-4 batch_size=128 weight_decay=0 \\
                   conv_dropout=0.1 dense_dropout=0 dense_dims=[] \\
                   block_sizes=[256,256,256,256] block_class=eff \\
    --output_dir results/snv_eval/legnet_k562_chrfold

This writes a configs.json + invokes inprocess_runner.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


# 10 fold val chromosomes (exclude chr 7+13 which are held as SNV test set).
DEFAULT_VAL_CHRS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11]


def build_configs(
    arch: str,
    n_folds: int,
    cell: str,
    hp_overrides: list[str],
    output_dir: Path,
    epochs: int,
    patience: int,
    aug: str,
    val_chrs: list[int],
    test_chrs: list[int],
    d_train: int = 0,
    label_source: str = "ag_oracle",
) -> list[dict]:
    """Build per-fold configs. Each fold differs only in val_chrs (one
    chromosome). test_chrs is shared (chr 7+13 always). For bar-plot
    comparisons use label_source='real' (K562_log2FC); for scaling-law
    / HP-search use label_source='ag_oracle' (denoised pseudolabels)."""
    configs = []
    for i in range(n_folds):
        val_chr = val_chrs[i % len(val_chrs)]
        label = f"fold{i:02d}_val_chr{val_chr}"
        cfg = {
            "label": label,
            "arch": arch,
            "d_train": d_train,  # 0 = full pool (minus val + test)
            "seed": 42 + i,
            "epochs": epochs,
            "patience": patience,
            "aug": aug,
            "label_source": label_source,
            "val_chrs": str(val_chr),
            "test_chrs": ",".join(str(c) for c in test_chrs),
            "output_dir": str(output_dir / label),
            "hp_overrides": list(hp_overrides),
        }
        configs.append(cfg)
    return configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arch",
        required=True,
        choices=["legnet", "dream_rnn", "dream_attn"],
        help="Currently only small arches use the in-process trainer. "
        "Foundation models use separate SLURM scripts.",
    )
    ap.add_argument("--cell", default="k562", choices=["k562", "hepg2", "sknsh"])
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--hp_overrides", nargs="*", default=[])
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument(
        "--aug",
        default="rev_complement",
        choices=["rev_complement", "rc_shift", "rc_shift_evoaug", "none"],
    )
    ap.add_argument(
        "--val_chrs",
        default=",".join(str(c) for c in DEFAULT_VAL_CHRS),
        help="Comma-sep val chrs (one per fold). Default = 10 chrs excluding 7,13.",
    )
    ap.add_argument(
        "--dry_run", action="store_true", help="Only write configs.json; do not invoke training."
    )
    ap.add_argument(
        "--k_parallel",
        type=int,
        default=4,
        help="parallel_gpu_runner trials concurrent per GPU. Small models can do 4-8.",
    )
    ap.add_argument(
        "--label_source",
        default="ag_oracle",
        choices=["ag_oracle", "real"],
        help="ag_oracle = AG-oracle pseudolabels (denoised; for scaling/HP search). "
        "real = K562_log2FC raw measurements (for bar-plot model comparisons).",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_chrs = [int(x) for x in args.val_chrs.split(",")]

    configs = build_configs(
        args.arch,
        args.n_folds,
        args.cell,
        args.hp_overrides,
        out_dir,
        args.epochs,
        args.patience,
        args.aug,
        val_chrs,
        test_chrs=[7, 13],
        d_train=0,
        label_source=args.label_source,
    )
    configs_path = out_dir / "configs.json"
    configs_path.write_text(json.dumps(configs, indent=2))
    print(f"Wrote {len(configs)} fold configs → {configs_path}")
    for c in configs:
        print(f"  {c['label']:25s} hp_overrides[:3]={c['hp_overrides'][:3]}")

    if args.dry_run:
        return

    # Each fold trains with different val_chrs, so data layouts differ per
    # config — must use the subprocess-style parallel_gpu_runner (in-process
    # trainer requires homogeneous data across configs).
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-u",
        str(REPO / "scripts/preflight/parallel_gpu_runner.py"),
        str(configs_path),
        str(args.k_parallel),
    ]
    print()
    print(f"Invoking parallel_gpu_runner k={args.k_parallel}: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(REPO))
    raise SystemExit(ret.returncode)


if __name__ == "__main__":
    main()
