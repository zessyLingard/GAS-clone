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
    0.6766966067864272,
    0.71248,
    0.730808383233533,
    0.75268,
    0.752,
    0.7645180722891567,
    0.7672535211267606,
    0.7680645161290323,
    0.7751818181818182,
    0.7874,
    0.7861111111111111,
    0.7935365853658537,
    0.7968421052631579,
    0.8052857142857143,
    0.7871212121212121,
    0.790483870967742,
    0.8008620689655173,
    0.7890740740740741,
    0.8080769230769231,
    0.8122
])

ratm_auc = np.array([
    0.5518172888015717,
    0.5520866141732284,
    0.5357100591715976,
    0.5411417322834645,
    0.5403465346534654,
    0.5339285714285714,
    0.5422222222222222,
    0.5472222222222223,
    0.5396428571428571,
    0.5438000000000001,
    0.5526086956521739,
    0.5480952380952381,
    0.5503846153846154,
    0.5422222222222222,
    0.5456060606060606,
    0.5369354838709678,
    0.5401724137931034,
    0.5485714285714286,
    0.5288461538461539,
    0.5302
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
        label="Fixed-IPD"
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
    "Fuzziness injection",
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