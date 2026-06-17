#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# Data
# =========================

sample_lengths = np.arange(100, 2001, 100)

original_auc = np.ones_like(sample_lengths, dtype=float)

fuzzy_auc = np.array([
    0.7772954091816368,
    0.83048,
    0.8597305389221557,
    0.8804399999999999,
    0.8881,
    0.8964457831325301,
    0.9013380281690141,
    0.9075806451612903,
    0.9128181818181819,
    0.9141999999999999,
    0.9154444444444445,
    0.9208536585365854,
    0.9218421052631578,
    0.9195714285714285,
    0.9240909090909091,
    0.9246774193548387,
    0.925,
    0.9294444444444445,
    0.9319230769230769,
    0.9302000000000001
])

ratm_auc = np.array([
    0.5524656188605108,
    0.5525590551181101,
    0.5535798816568047,
    0.5472834645669292,
    0.5587623762376238,
    0.5552380952380953,
    0.548888888888889,
    0.555,
    0.5589285714285714,
    0.5688,
    0.5721739130434782,
    0.5702380952380952,
    0.5829487179487179,
    0.5805555555555555,
    0.5910606060606061,
    0.5740322580645161,
    0.575,
    0.5796428571428571,
    0.5834615384615385,
    0.575
])


# =========================
# Style
# =========================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 17,
    "axes.labelsize": 20,
    "legend.fontsize": 17,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#4C72B0"
ORANGE = "#DD8452"


# =========================
# Plot function
# =========================

def plot_gas_auc(method_auc, method_label, output_path):
    # Thesis-sized figure
    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    ax.plot(
        sample_lengths,
        original_auc,
        color=BLUE,
        marker="o",
        markersize=5.2,
        linewidth=2.4,
        label="Original"
    )

    ax.plot(
        sample_lengths,
        method_auc,
        color=ORANGE,
        marker="s",
        markersize=5.2,
        linewidth=2.4,
        label=method_label
    )

    ax.set_xlabel("Sample length (IPDs)")
    ax.set_ylabel("GAS AUC")

    ax.set_xlim(100, 2000)
    ax.set_ylim(0.5, 1.02)

    # Major labels: readable
    ax.set_xticks([100, 500, 1000, 1500, 2000])

    # Minor ticks: shows that measurements are every 100 IPDs
    ax.set_xticks(sample_lengths, minor=True)
    ax.tick_params(axis="x", which="minor", length=4, width=1.0)
    ax.tick_params(axis="x", which="major", length=7, width=1.1)
    ax.tick_params(axis="y", which="major", length=7, width=1.1)

    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    ax.grid(True, which="major", linewidth=0.8, alpha=0.35)

    # Legend outside the plot area
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False
    )

    fig.tight_layout()

    output_path = Path(output_path)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# =========================
# Save figures
# =========================

out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

plot_gas_auc(
    fuzzy_auc,
    "Fuzzy injection",
    out_dir / "gas_auc_fuzzy"
)

plot_gas_auc(
    ratm_auc,
    "RATM",
    out_dir / "gas_auc_ratm"
)

print("Saved:")
print("  figures/gas_auc_fuzzy.pdf")
print("  figures/gas_auc_fuzzy.png")
print("  figures/gas_auc_ratm.pdf")
print("  figures/gas_auc_ratm.png")