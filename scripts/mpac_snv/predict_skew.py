"""MPAC-style allelic-skew prediction with sliding-window + strand averaging.

Computes the allelic skew (alt_activity - ref_activity) for SNV pairs using
the protocol from Butts et al. (bioRxiv 2025.04.16):

  For each (ref_seq, alt_seq):
    1. Slide an N-window over the 200-bp model input (step_size = stride bp).
    2. For each window position, predict on BOTH strands (forward + RC).
    3. Average across (windows × strands) → single ref_act and alt_act.
    4. skew = alt_act - ref_act.

For each model in `model_paths`, do the above, then average across the
ensemble (each model's ref/alt activity averaged before computing skew).

Defaults match MPAC: 18 windows × 2 strands (positions 9 to 181 in 10-bp
steps). For our 200-bp K562 sequences with no extra flanking, we fall back
to a reduced sliding scheme (3 windows × 2 strands) controlled by
`--n_windows` and `--step`.

Usage:
  python -m scripts.preflight.snv_eval.predict_skew \\
    --snv_parquet outputs/oracle_pseudolabels_k562_ag_s2_refalt/pool/snv_pairs.parquet \\
    --chrs 7,13 \\
    --models <checkpoint_dir_1> <checkpoint_dir_2> ... \\
    --arch legnet \\
    --output_dir results/snv_eval/k562_chr7_13_mpac
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]


_NUC = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seqs: list[str], seq_len: int = 200, in_channels: int = 4) -> torch.Tensor:
    """One-hot encode (B, C=4, L). Center-pads with N when short."""
    n = len(seqs)
    out = np.zeros((n, in_channels, seq_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s.upper()
        L = len(s)
        pad = (seq_len - L) // 2  # center
        for j, nt in enumerate(s):
            if nt in _NUC:
                pos = j + pad
                if 0 <= pos < seq_len:
                    out[i, _NUC[nt], pos] = 1.0
    return torch.from_numpy(out)


def _rc(x: torch.Tensor) -> torch.Tensor:
    """Reverse-complement one-hot batch (B, 4, L). ACGT → TGCA via channel reverse + L reverse."""
    return x.flip(dims=(1, 2))


def _slide(x: torch.Tensor, n_windows: int, step: int, seq_len: int) -> list[torch.Tensor]:
    """Return N shifted views of x. Step=0 returns just x."""
    if n_windows <= 1 or step == 0:
        return [x]
    center = (n_windows - 1) // 2
    out = []
    for w in range(n_windows):
        offset = (w - center) * step
        if offset == 0:
            out.append(x)
        else:
            # circular shift along length axis. (We avoid index gymnastics — if
            # adapter padding is present, this shifts the variant within the
            # 200-bp window.)
            out.append(torch.roll(x, shifts=offset, dims=2))
    return out


def _build_legnet(state_dict: dict, hp: dict) -> torch.nn.Module:
    from models.legnet import LegNet

    m = LegNet(
        in_channels=hp.get("in_channels", 4),
        block_sizes=hp.get("block_sizes"),
        ks=hp.get("ks", 5),
        conv_dropout=hp.get("conv_dropout", hp.get("dropout", 0.0)),
        dense_dims=hp.get("dense_dims", []) or [],
        dense_dropout=hp.get("dense_dropout", 0.0),
        block_class=hp.get("block_class", "eff"),
        task_mode="k562",
    )
    m.load_state_dict(state_dict, strict=False)
    return m


def _load_checkpoint(path: Path, arch: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    hp = ckpt.get("hp") or ckpt.get("model_info") or {}
    if arch == "legnet":
        m = _build_legnet(state, hp).to(device)
    else:
        raise NotImplementedError(
            f"Loading arch={arch} not yet implemented in predict_skew. "
            "Add the arch-specific build path here."
        )
    m.eval()
    return m


@torch.inference_mode()
def predict_activity(
    model: torch.nn.Module,
    seqs: list[str],
    device: torch.device,
    seq_len: int = 200,
    in_channels: int = 4,
    n_windows: int = 3,
    step: int = 10,
    batch_size: int = 1024,
    use_bf16: bool = True,
) -> np.ndarray:
    """Predict scalar activity per sequence with sliding-window + RC averaging.

    Speedups: bf16 autocast on Ampere+ GPUs (~1.7× over fp32, no scaler needed);
    default batch_size raised to 1024 (~2-4× over BS=256 for small models)."""
    X = one_hot(seqs, seq_len=seq_len, in_channels=in_channels)
    preds_acc = torch.zeros(len(seqs), dtype=torch.float64)
    n_avg = 0

    amp_ok = use_bf16 and device.type == "cuda"
    if amp_ok:
        try:
            cap = torch.cuda.get_device_capability(0)
            amp_ok = cap[0] >= 8  # Ampere or later
        except Exception:
            amp_ok = False

    # Iterate over (window-shifted, strand) pairs; for each, run batched fwd.
    for window_view in _slide(X, n_windows, step, seq_len):
        for view in (window_view, _rc(window_view)):
            preds = []
            for s in range(0, len(view), batch_size):
                xb = view[s : s + batch_size].to(device, non_blocking=True)
                if amp_ok:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        yhat = model(xb).reshape(-1).float().cpu()
                else:
                    yhat = model(xb).reshape(-1).float().cpu()
                preds.append(yhat)
            preds = torch.cat(preds).double()
            preds_acc += preds
            n_avg += 1

    return (preds_acc / n_avg).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--snv_parquet",
        required=True,
        help="Parquet with cols: locus, sequence_ref, sequence_alt, ref_log2FC, alt_log2FC.",
    )
    ap.add_argument(
        "--chrs",
        default="7,13",
        help="Comma-separated chromosomes to filter (matches the prefix of `locus`).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more checkpoint files (.pt). Predictions are ensembled (mean).",
    )
    ap.add_argument("--arch", default="legnet")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seq_len", type=int, default=200)
    ap.add_argument("--in_channels", type=int, default=4)
    ap.add_argument(
        "--n_windows",
        type=int,
        default=3,
        help="Number of sliding windows per variant. Use 18 with --step 10 to match MPAC.",
    )
    ap.add_argument("--step", type=int, default=10, help="Stride between sliding windows in bp.")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument(
        "--bf16", action="store_true", default=True, help="bf16 autocast on Ampere+ (default on)."
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_parquet(args.snv_parquet)
    chr_filter = [c.strip() for c in args.chrs.split(",")]
    chr_filter_set = set(chr_filter)
    df["chr"] = df["locus"].str.split(":").str[0]
    df = df[df["chr"].isin(chr_filter_set)].reset_index(drop=True)
    print(f"Filtered to {len(df):,} SNV pairs on chr {','.join(chr_filter)}")

    ref_seqs = df["sequence_ref"].tolist()
    alt_seqs = df["sequence_alt"].tolist()
    n = len(df)

    ref_acc = np.zeros(n, dtype=np.float64)
    alt_acc = np.zeros(n, dtype=np.float64)
    per_model_skews = []
    n_models = 0

    for mpath in args.models:
        t0 = time.time()
        model = _load_checkpoint(Path(mpath), args.arch, device)
        ref_act = predict_activity(
            model,
            ref_seqs,
            device,
            args.seq_len,
            args.in_channels,
            args.n_windows,
            args.step,
            args.batch_size,
            use_bf16=args.bf16,
        )
        alt_act = predict_activity(
            model,
            alt_seqs,
            device,
            args.seq_len,
            args.in_channels,
            args.n_windows,
            args.step,
            args.batch_size,
            use_bf16=args.bf16,
        )
        skew_m = alt_act - ref_act
        per_model_skews.append(skew_m)
        ref_acc += ref_act
        alt_acc += alt_act
        n_models += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Per-model spot Pearson vs empirical
        empirical_skew = (df["alt_log2FC"] - df["ref_log2FC"]).to_numpy()
        valid = np.isfinite(empirical_skew) & np.isfinite(skew_m)
        if valid.sum() > 1:
            r = float(np.corrcoef(skew_m[valid], empirical_skew[valid])[0, 1])
            print(
                f"  [{Path(mpath).name}] n_models={n_models}  per-model Pearson r={r:.4f}  ({time.time() - t0:.1f}s)"
            )

    # Ensemble: mean activity across models
    ref_mean = ref_acc / max(1, n_models)
    alt_mean = alt_acc / max(1, n_models)
    skew_ensemble = alt_mean - ref_mean

    empirical_skew = (df["alt_log2FC"] - df["ref_log2FC"]).to_numpy()
    empirical_ref = df["ref_log2FC"].to_numpy()
    empirical_alt = df["alt_log2FC"].to_numpy()

    def _r_mse(pred, emp):
        v = np.isfinite(pred) & np.isfinite(emp)
        if v.sum() < 2:
            return float("nan"), float("nan"), int(v.sum())
        return (
            float(np.corrcoef(pred[v], emp[v])[0, 1]),
            float(np.mean((pred[v] - emp[v]) ** 2)),
            int(v.sum()),
        )

    # Three bar-plot metrics on the held-out chr 7+13 SNV pool:
    r_ref, mse_ref, n_ref = _r_mse(ref_mean, empirical_ref)
    r_alt, mse_alt, n_alt = _r_mse(alt_mean, empirical_alt)
    r_skew, mse_skew, n_skew = _r_mse(skew_ensemble, empirical_skew)

    print()
    print(f"=== ENSEMBLE ({n_models} models, n_windows={args.n_windows}, step={args.step}):")
    print(f"  reference activity   Pearson r = {r_ref:.5f}   MSE = {mse_ref:.5f}  n = {n_ref:,}")
    print(f"  alternate allele     Pearson r = {r_alt:.5f}   MSE = {mse_alt:.5f}  n = {n_alt:,}")
    print(f"  SNV effect (skew)    Pearson r = {r_skew:.5f}   MSE = {mse_skew:.5f}  n = {n_skew:,}")

    out_df = df[["locus", "chr"]].copy()
    out_df["ref_activity_pred"] = ref_mean
    out_df["alt_activity_pred"] = alt_mean
    out_df["skew_pred"] = skew_ensemble
    out_df["skew_empirical"] = empirical_skew
    out_df["ref_empirical"] = empirical_ref
    out_df["alt_empirical"] = empirical_alt
    out_df.to_parquet(out_dir / "skew_predictions.parquet", index=False)

    valid_skew = np.isfinite(empirical_skew) & np.isfinite(skew_ensemble)
    summary = {
        "arch": args.arch,
        "n_models": n_models,
        "n_windows": args.n_windows,
        "step_bp": args.step,
        "chrs": chr_filter,
        "n_variants": n_skew,
        # Three bar-plot metrics (Pearson r vs raw K562_log2FC measurements).
        "ref_pearson": r_ref,
        "ref_mse": mse_ref,
        "alt_pearson": r_alt,
        "alt_mse": mse_alt,
        "skew_pearson": r_skew,
        "skew_mse": mse_skew,
        # Back-compat aliases
        "ensemble_pearson_skew": r_skew,
        "ensemble_mse_skew": mse_skew,
        "per_model_pearson": [
            float(np.corrcoef(s[valid_skew], empirical_skew[valid_skew])[0, 1])
            for s in per_model_skews
        ],
        "model_paths": [str(p) for p in args.models],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir}/skew_predictions.parquet + summary.json")


if __name__ == "__main__":
    main()
