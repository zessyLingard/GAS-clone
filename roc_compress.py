#!/usr/bin/env python3
"""
roc_compress.py

Compute Cabuk/Wendzel-style compressibility scores and AUC
for one legitimate IPD CSV and one covert IPD CSV.

Outputs:
  1. compress_auc_summary.csv
  2. compress_scores_per_window.csv
  3. compress_score_stats.json
  4. compress_roc.pdf / compress_roc.png
  5. compress_score_hist.pdf / compress_score_hist.png
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


# =========================
# Compressibility functions
# =========================

def iat2str(i):
    """
    Cabuk/Wendzel-style IPD string conversion:
      - round to two significant digits
      - encode leading zeros after decimal as A/B/C...
      - concatenate remaining significant digits
    """
    if i == 0:
        return "0"

    try:
        i = round(float(i), 2 - int(math.floor(math.log10(abs(float(i))))) - 1)
    except (ValueError, OverflowError):
        return ""

    s = "{:.16f}".format(i).split(".")[1]
    leading_zeros = len(s) - len(s.lstrip("0"))

    stripped = s.strip("0")
    if stripped == "":
        stripped = "0"

    if leading_zeros == 0:
        return stripped

    return chr(64 + leading_zeros) + stripped


def compress_score_series(x):
    """
    Compute one compressibility score for one IPD window.
    """
    mystring = x.apply(iat2str)
    full_string = mystring.str.cat()

    if not full_string:
        return np.nan

    original_size = len(full_string.encode())
    compressed_size = len(gzip.compress(full_string.encode()))

    if compressed_size == 0:
        return np.nan

    return original_size / compressed_size


# =========================
# Data loading
# =========================

def read_ipds(path, unit="s"):
    """
    Read IPDs from CSV and return values in seconds.
    Supports:
      - IPDs header
      - IPD header
      - transmitted_ms header
      - no header
    """
    path = Path(path)
    df = pd.read_csv(path)

    if "IPDs" in df.columns:
        vals = pd.to_numeric(df["IPDs"], errors="coerce")
    elif "IPD" in df.columns:
        vals = pd.to_numeric(df["IPD"], errors="coerce")
    elif "transmitted_ms" in df.columns:
        vals = pd.to_numeric(df["transmitted_ms"], errors="coerce")
        unit = "ms"
    else:
        df = pd.read_csv(path, header=None)
        vals = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    vals = vals.dropna().astype(float)

    if unit == "ms":
        vals = vals / 1000.0
    elif unit == "auto":
        med = float(vals.median()) if len(vals) else 0.0
        if med > 2.0:
            print(f"[auto-unit] {path}: median={med:.6f}, assuming milliseconds")
            vals = vals / 1000.0
        else:
            print(f"[auto-unit] {path}: median={med:.6f}, assuming seconds")
    elif unit == "s":
        pass
    else:
        raise ValueError("unit must be s, ms, or auto")

    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]

    return vals.reset_index(drop=True)


def make_window_scores(vals, window_size):
    """
    Drop IPDs > 1.0 second and split into complete windows.
    """
    before_drop = len(vals)

    vals = vals[vals <= 1.0].reset_index(drop=True)
    after_drop = len(vals)

    n_windows = len(vals) // window_size
    if n_windows < 1:
        return pd.Series(dtype=float), before_drop, after_drop

    usable = vals.iloc[:n_windows * window_size]
    groups = np.arange(len(usable)) // window_size

    scores = usable.groupby(groups).apply(compress_score_series)
    scores = scores.dropna().astype(float).reset_index(drop=True)

    return scores, before_drop, after_drop


def summarize_scores(name, scores):
    if len(scores) == 0:
        return {"name": name, "count": 0}

    desc = scores.describe()

    return {
        "name": name,
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if not pd.isna(desc["std"]) else 0.0,
        "min": float(desc["min"]),
        "25%": float(desc["25%"]),
        "50%": float(desc["50%"]),
        "75%": float(desc["75%"]),
        "max": float(desc["max"]),
    }


# =========================
# Plot style
# =========================

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 17,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# =========================
# Plotting
# =========================

def plot_roc(out, fpr, tpr, auc_value):
    BLUE = "#4C72B0"
    ORANGE = "#DD8452"

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    ax.plot(
        fpr,
        tpr,
        drawstyle="steps-post",
        linewidth=2.2,
        color=ORANGE,
        label=f"AUC = {auc_value:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=BLUE,
        linewidth=1.6,
        label="Random classifier"
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel("False positive rate", labelpad=10)
    ax.set_ylabel("True positive rate", labelpad=10)

    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.grid(True, linewidth=0.6, alpha=0.30)

    ax.legend(
        loc="lower right",
        frameon=True,
        borderpad=0.5,
        handlelength=2.5
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.96,
        bottom=0.16,
        top=0.96
    )

    fig.savefig(out / "compress_roc.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out / "compress_roc.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_score_line(out, legit_scores, covert_scores, method_name):
    BLUE = "#4C72B0"
    ORANGE = "#DD8452"

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    # Sử dụng KDE plot của pandas để vẽ đường phân phối
    legit_scores.plot.kde(
        ax=ax, 
        color=BLUE, 
        linewidth=2.5, 
        label="Legit VPN traffic"
    )
    
    covert_scores.plot.kde(
        ax=ax, 
        color=ORANGE, 
        linewidth=2.5, 
        label=method_name
    )

    ax.set_xlabel("Compressibility score", labelpad=10)
    ax.set_ylabel("Density", labelpad=10) # Đổi thành Density khi dùng KDE

    ax.grid(True, linewidth=0.6, alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.96,
        bottom=0.16,
        top=0.88
    )

    # Lưu với tên file mới để phân biệt
    fig.savefig(out / "compress_score_line.pdf", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out / "compress_score_line.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--legit", required=True, help="Legitimate IPD CSV")
    ap.add_argument("--covert", required=True, help="Covert IPD CSV")
    ap.add_argument("--name", default="Covert", help="Covert method name")
    ap.add_argument("--out", default="compress_out", help="Output directory")
    ap.add_argument("--window", type=int, default=510, help="Window size in IPDs")
    ap.add_argument("--unit", choices=["s", "ms", "auto"], default="s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shuffle", action="store_true")

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    legit = read_ipds(args.legit, unit=args.unit)
    covert = read_ipds(args.covert, unit=args.unit)

    legit_scores, legit_before, legit_after = make_window_scores(legit, args.window)
    covert_scores, covert_before, covert_after = make_window_scores(covert, args.window)

    n = min(len(legit_scores), len(covert_scores))

    if n < 2:
        raise SystemExit(
            "ERROR: Not enough complete windows after dropping IPDs > 1.0 sec.\n"
            f"legit windows={len(legit_scores)}, covert windows={len(covert_scores)}, window={args.window}\n"
            f"legit IPDs after drop={legit_after}, covert IPDs after drop={covert_after}"
        )

    legit_scores = legit_scores.iloc[:n].reset_index(drop=True)
    covert_scores = covert_scores.iloc[:n].reset_index(drop=True)

    y_true = np.array([0] * n + [1] * n, dtype=int)
    y_score = np.concatenate([legit_scores.values, covert_scores.values])

    if not args.no_shuffle:
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(y_true))
        y_true = y_true[order]
        y_score = y_score[order]

    raw_auc = float(roc_auc_score(y_true, y_score))
    corrected_auc = float(max(raw_auc, 1.0 - raw_auc))

    if raw_auc < 0.5:
        fpr, tpr, thresholds = roc_curve(y_true, -y_score)
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

    if fpr[0] != 0.0 or tpr[0] != 0.0:
        fpr = np.r_[0.0, fpr]
        tpr = np.r_[0.0, tpr]
        thresholds = np.r_[np.inf, thresholds]

    stats = {
        "legit": summarize_scores("legit", legit_scores),
        args.name: summarize_scores(args.name, covert_scores),
    }

    summary = {
        "method": args.name,
        "legit_csv": args.legit,
        "covert_csv": args.covert,
        "unit": args.unit,
        "window_size": args.window,
        "legit_ipds_before_drop": int(legit_before),
        "covert_ipds_before_drop": int(covert_before),
        "legit_ipds_after_drop_gt_1s": int(legit_after),
        "covert_ipds_after_drop_gt_1s": int(covert_after),
        "used_windows_per_class": int(n),
        "mixed_total_windows": int(2 * n),
        "compressibility_auc_raw": raw_auc,
        "compressibility_auc_corrected": corrected_auc,
        "legit_score_mean": stats["legit"]["mean"],
        "legit_score_std": stats["legit"]["std"],
        f"{args.name}_score_mean": stats[args.name]["mean"],
        f"{args.name}_score_std": stats[args.name]["std"],
    }

    # Save summary
    pd.DataFrame([summary]).to_csv(out / "compress_auc_summary.csv", index=False)

    # Save per-window scores
    rows = []

    for i, s in enumerate(legit_scores):
        rows.append({
            "source": "legit",
            "label": 0,
            "window_id": i,
            "compressibility_score": float(s),
        })

    for i, s in enumerate(covert_scores):
        rows.append({
            "source": args.name,
            "label": 1,
            "window_id": i,
            "compressibility_score": float(s),
        })

    pd.DataFrame(rows).to_csv(out / "compress_scores_per_window.csv", index=False)

    # Save score stats
    (out / "compress_score_stats.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )

    # Save ROC points
    pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds
    }).to_csv(out / "compress_roc_points.csv", index=False)

    # Save figures
    set_plot_style()
    plot_roc(out, fpr, tpr, corrected_auc)
    plot_score_line(out, legit_scores, covert_scores, args.name)

    print("=== Compressibility result ===")
    print(f"legit file              : {args.legit}")
    print(f"covert file             : {args.covert}")
    print(f"method                  : {args.name}")
    print(f"window size             : {args.window}")
    print(f"used windows/class      : {n}")
    print(f"raw AUC                 : {raw_auc:.6f}")
    print(f"corrected AUC           : {corrected_auc:.6f}")
    print()
    print("Score stats:")
    print(pd.DataFrame([stats["legit"], stats[args.name]]).to_string(index=False))
    print()
    print("Saved:")
    print(f"  {out / 'compress_auc_summary.csv'}")
    print(f"  {out / 'compress_scores_per_window.csv'}")
    print(f"  {out / 'compress_score_stats.json'}")
    print(f"  {out / 'compress_roc_points.csv'}")
    print(f"  {out / 'compress_roc.pdf'}")
    print(f"  {out / 'compress_roc.png'}")
    print(f"  {out / 'compress_score_hist.pdf'}")
    print(f"  {out / 'compress_score_hist.png'}")


if __name__ == "__main__":
    main()