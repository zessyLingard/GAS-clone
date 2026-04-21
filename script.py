import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def resolve_ipd_column(df: pd.DataFrame) -> str:
    preferred = ["IPD", "IPDs", "ipd", "ipds", "inter_packet_delay"]
    for col in preferred:
        if col in df.columns:
            return col

    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if numeric_cols:
        return numeric_cols[0]

    return df.columns[0]


def ensure_output_dirs(base_dir: Path) -> tuple[Path, Path]:
    report_dir = base_dir / "exp" / "eda"
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return report_dir, plot_dir


def calculate_quality_metrics(raw_series: pd.Series, cleaned_series: pd.Series) -> dict:
    total_rows = len(raw_series)
    missing_rows = raw_series.isna().sum()
    non_numeric_rows = raw_series.notna().sum() - cleaned_series.shape[0]
    zero_rows = int((cleaned_series == 0).sum())
    negative_rows = int((cleaned_series < 0).sum())

    return {
        "total_rows": int(total_rows),
        "missing_rows": int(missing_rows),
        "missing_ratio_pct": (missing_rows / total_rows * 100) if total_rows else 0.0,
        "non_numeric_rows": int(max(non_numeric_rows, 0)),
        "zero_rows": zero_rows,
        "zero_ratio_pct": (zero_rows / len(cleaned_series) * 100) if len(cleaned_series) else 0.0,
        "negative_rows": negative_rows,
        "negative_ratio_pct": (negative_rows / len(cleaned_series) * 100) if len(cleaned_series) else 0.0,
    }


def calculate_distribution_metrics(ipd: pd.Series) -> dict:
    if ipd.empty:
        return {}

    q = ipd.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    iqr = q.loc[0.75] - q.loc[0.25]
    cv = (ipd.std(ddof=1) / ipd.mean()) if ipd.mean() != 0 else np.nan

    return {
        "count": int(ipd.shape[0]),
        "min": float(ipd.min()),
        "max": float(ipd.max()),
        "mean": float(ipd.mean()),
        "median": float(q.loc[0.5]),
        "std": float(ipd.std(ddof=1)),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "iqr": float(iqr),
        "cv": float(cv) if np.isfinite(cv) else np.nan,
        "skewness": float(skew(ipd, bias=False, nan_policy="omit")),
        "kurtosis": float(kurtosis(ipd, bias=False, nan_policy="omit")),
        "acf_lag1": float(ipd.autocorr(lag=1)),
    }


def save_interval_breakdown(ipd: pd.Series, report_dir: Path) -> pd.DataFrame:
    intervals = [
        ("burst_jitter", 0.0, 0.1),
        ("idle_start", 0.1, 0.4),
        ("bit_0_zone", 0.4, 0.6),
        ("safe_gap", 0.6, 0.8),
        ("bit_1_zone", 0.8, 1.2),
        ("long_tail", 1.2, 5.2),
    ]

    rows = []
    for label, low, high in intervals:
        count = int(((ipd >= low) & (ipd < high)).sum())
        ratio = (count / len(ipd) * 100) if len(ipd) else 0.0
        rows.append({"interval": label, "low": low, "high": high, "count": count, "ratio_pct": ratio})

    interval_df = pd.DataFrame(rows)
    interval_df.to_csv(report_dir / "interval_breakdown.csv", index=False)
    return interval_df


def plot_histograms(ipd: pd.Series, log_ipd: pd.Series, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(ipd, bins=80, color="#1f77b4", alpha=0.85)
    axes[0].set_title("IPD Histogram")
    axes[0].set_xlabel("IPD")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(log_ipd, bins=80, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("log(IPD + 1e-9) Histogram")
    axes[1].set_xlabel("log(IPD + 1e-9)")
    axes[1].set_ylabel("Frequency")

    fig.tight_layout()
    fig.savefig(plot_dir / "histograms.png", dpi=160)
    plt.close(fig)


def plot_boxplot(ipd: pd.Series, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(ipd.values, vert=True, showfliers=True)
    ax.set_title("IPD Boxplot")
    ax.set_ylabel("IPD")
    fig.tight_layout()
    fig.savefig(plot_dir / "boxplot.png", dpi=160)
    plt.close(fig)


def select_timeseries_points(ipd: pd.Series, max_points: int, strategy: str) -> pd.Series:
    if max_points <= 0 or max_points >= len(ipd):
        return ipd

    if strategy == "first":
        return ipd.iloc[:max_points]

    # Uniform sampling keeps global structure when the series is very long.
    indices = np.linspace(0, len(ipd) - 1, num=max_points, dtype=int)
    return ipd.iloc[indices]


def plot_timeseries(ipd: pd.Series, plot_dir: Path, max_points: int = 0, strategy: str = "uniform") -> None:
    sampled = select_timeseries_points(ipd, max_points=max_points, strategy=strategy)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sampled.index, sampled.values, linewidth=0.8, color="#2ca02c")
    ax.set_title(
        f"IPD Time Series ({len(sampled)} / {len(ipd)} points, strategy={strategy})"
    )
    ax.set_xlabel("Packet index")
    ax.set_ylabel("IPD")
    fig.tight_layout()
    fig.savefig(plot_dir / "timeseries.png", dpi=160)
    plt.close(fig)


def plot_ecdf(ipd: pd.Series, plot_dir: Path) -> None:
    x = np.sort(ipd.values)
    y = np.arange(1, len(x) + 1) / len(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color="#d62728", linewidth=1.2)
    ax.set_title("IPD ECDF")
    ax.set_xlabel("IPD")
    ax.set_ylabel("F(x)")
    fig.tight_layout()
    fig.savefig(plot_dir / "ecdf.png", dpi=160)
    plt.close(fig)


def save_feature_summary(ipd: pd.Series, report_dir: Path) -> Path:
    q = ipd.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    small_threshold = q.loc[0.1]
    large_threshold = q.loc[0.9]

    feature_row = {
        "n_packets": int(len(ipd)),
        "mean": float(ipd.mean()),
        "median": float(q.loc[0.5]),
        "std": float(ipd.std(ddof=1)),
        "iqr": float(q.loc[0.75] - q.loc[0.25]),
        "p90": float(q.loc[0.9]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "max": float(ipd.max()),
        "skewness": float(skew(ipd, bias=False, nan_policy="omit")),
        "kurtosis": float(kurtosis(ipd, bias=False, nan_policy="omit")),
        "acf_lag1": float(ipd.autocorr(lag=1)),
        "burst_ratio": float((ipd <= small_threshold).mean()),
        "silence_ratio": float((ipd >= large_threshold).mean()),
    }

    out_path = report_dir / "feature_summary.csv"
    pd.DataFrame([feature_row]).to_csv(out_path, index=False)
    return out_path


def write_report(report_path: Path, data_path: Path, ipd_col: str, quality: dict, dist: dict) -> None:
    lines = [
        "IPD EDA REPORT",
        "=" * 60,
        f"input_file: {data_path}",
        f"ipd_column: {ipd_col}",
        "",
        "[Data Quality]",
    ]

    for key, value in quality.items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("[Distribution Stats]")
    for key, value in dist.items():
        lines.append(f"- {key}: {value}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for traffic inter-packet delay (IPD) CSV.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/labnet/timestamps.csv",
        help="Path to CSV file containing IPD values.",
    )
    parser.add_argument(
        "--timeseries-points",
        type=int,
        default=20000,
        help="Number of points in timeseries plot (0 = use all points).",
    )
    parser.add_argument(
        "--timeseries-strategy",
        type=str,
        choices=["uniform", "first"],
        default="uniform",
        help="Point selection strategy when --timeseries-points is smaller than dataset size.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent
    data_path = (root_dir / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    report_dir, plot_dir = ensure_output_dirs(root_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    ipd_col = resolve_ipd_column(df)
    raw = df[ipd_col]
    ipd = pd.to_numeric(raw, errors="coerce").dropna()

    if ipd.empty:
        raise ValueError(f"Column '{ipd_col}' has no valid numeric values.")

    quality_metrics = calculate_quality_metrics(raw, ipd)
    dist_metrics = calculate_distribution_metrics(ipd)

    interval_df = save_interval_breakdown(ipd, report_dir)
    log_ipd = np.log(ipd + 1e-9)

    plot_histograms(ipd, log_ipd, plot_dir)
    plot_boxplot(ipd, plot_dir)
    plot_timeseries(
        ipd,
        plot_dir,
        max_points=args.timeseries_points,
        strategy=args.timeseries_strategy,
    )
    plot_ecdf(ipd, plot_dir)
    feature_path = save_feature_summary(ipd, report_dir)

    report_path = report_dir / "report.txt"
    write_report(report_path, data_path, ipd_col, quality_metrics, dist_metrics)

    print("IPD EDA completed")
    print(f"Input file: {data_path}")
    print(f"Detected IPD column: {ipd_col}")
    print(f"Rows (valid numeric IPD): {len(ipd)}")
    print(f"Report: {report_path}")
    print(f"Interval CSV: {report_dir / 'interval_breakdown.csv'}")
    print(f"Feature CSV: {feature_path}")
    print(f"Plots directory: {plot_dir}")
    print(
        f"Timeseries plot config: points={args.timeseries_points} (0 means all), "
        f"strategy={args.timeseries_strategy}"
    )
    print("\nInterval breakdown preview:")
    print(interval_df.to_string(index=False))


if __name__ == "__main__":
    main()