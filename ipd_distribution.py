#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


FILES = {
    "Fixed-IPD": "data/labnet/fixed_big.csv",
    "Fuzziness injection": "data/labnet/wendzel_tau150ms_big.csv",
    "RATM": "data/labnet/ratm.csv",
    "LEGIT": "data/bignet/vpn.csv",
    "HTTP": "data/bignet/http.csv"
}

OUT_DIR = Path("figures")

MAX_IPD = 0.45
BINS = 70
Y_MAX = 55


def read_ipds(path):
    df = pd.read_csv(path)

    if "IPDs" in df.columns:
        vals = pd.to_numeric(df["IPDs"], errors="coerce")
    elif "IPD" in df.columns:
        vals = pd.to_numeric(df["IPD"], errors="coerce")
    else:
        vals = pd.to_numeric(df.iloc[:, 0], errors="coerce")

    vals = vals.dropna().astype(float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    return vals.reset_index(drop=True)


def percent_weights(x):
    return np.ones(len(x), dtype=float) * 100.0 / len(x)


def safe_name(name):
    return name.lower().replace(" ", "_")


def plot_one(name, ipds, bins):
    shown = ipds[ipds <= MAX_IPD]

    # Thesis-sized figure
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    ax.hist(
        shown,
        bins=bins,
        weights=percent_weights(shown),
        alpha=0.82,
        edgecolor="black",
        linewidth=0.4,
        color="#4C72B0"
    )

    # Original timing centers
    ax.axvline(0.150, linestyle="--", linewidth=1.4, alpha=0.80)
    ax.axvline(0.300, linestyle="--", linewidth=1.4, alpha=0.80)

    ax.set_xlim(0, MAX_IPD)
    ax.set_ylim(0, Y_MAX)

    ax.set_xticks(np.arange(0.00, MAX_IPD + 0.001, 0.05))
    ax.set_yticks([0, 10, 20, 30, 40, 50])

    ax.set_xlabel("Inter-packet delay (s)")
    ax.set_ylabel("IPDs per bin (%)")

    ax.grid(True, linewidth=0.6, alpha=0.30)

    ax.text(
        0.97,
        0.90,
        name,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=18
    )

    ax.text(
        0.97,
        0.82,
        f"$n={len(shown)}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=16
    )

    fig.tight_layout()

    out_base = OUT_DIR / f"ipd_{safe_name(name)}"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    print(f"{name}: shown {len(shown)} / total {len(ipds)}")
    print(f"Saved {out_base.with_suffix('.pdf')}")
    print(f"Saved {out_base.with_suffix('.png')}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    data = {}
    for name, path in FILES.items():
        ipds = read_ipds(path)
        data[name] = ipds
        print(f"{name}: {len(ipds)} IPDs loaded from {path}")

    bins = np.linspace(0, MAX_IPD, BINS)

    for name, ipds in data.items():
        plot_one(name, ipds, bins)


if __name__ == "__main__":
    main()