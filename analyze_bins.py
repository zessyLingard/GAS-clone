from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_ipds(csv_path: Path) -> np.ndarray:
	df = pd.read_csv(csv_path, header=None)
	ipds = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=float)
	if ipds.size == 0:
		raise ValueError(f"No numeric IPD values found in {csv_path}")
	return ipds


def burst_lengths(ipds: np.ndarray, threshold_s: float) -> list[int]:
	active = ipds < threshold_s
	lengths: list[int] = []
	current = 0

	for is_active in active:
		if is_active:
			current += 1
		elif current:
			lengths.append(current)
			current = 0

	if current:
		lengths.append(current)

	return lengths


def autocorrelation_lag1(ipds: np.ndarray) -> float:
	if ipds.size < 2:
		return float("nan")
	x = ipds[:-1]
	y = ipds[1:]
	if np.std(x) == 0 or np.std(y) == 0:
		return float("nan")
	return float(np.corrcoef(x, y)[0, 1])


def summarize(ipds: np.ndarray, burst_threshold_ms: float, rounded_decimals: int = 6) -> str:
	ipds_ms = ipds * 1000.0
	burst_threshold_s = burst_threshold_ms / 1000.0
	summary_lines = []

	summary_lines.append("Covert traffic IPD analysis")
	summary_lines.append(f"Samples: {ipds.size}")
	summary_lines.append("")
	summary_lines.append("Descriptive statistics (ms)")
	summary_lines.append(f"Mean: {np.mean(ipds_ms):.6f}")
	summary_lines.append(f"Std: {np.std(ipds_ms, ddof=1):.6f}")
	summary_lines.append(f"Min: {np.min(ipds_ms):.6f}")
	summary_lines.append(f"Q25: {np.percentile(ipds_ms, 25):.6f}")
	summary_lines.append(f"Median: {np.median(ipds_ms):.6f}")
	summary_lines.append(f"Q75: {np.percentile(ipds_ms, 75):.6f}")
	summary_lines.append(f"Q95: {np.percentile(ipds_ms, 95):.6f}")
	summary_lines.append(f"Max: {np.max(ipds_ms):.6f}")
	summary_lines.append(f"Lag-1 autocorrelation: {autocorrelation_lag1(ipds):.6f}")
	summary_lines.append("")
	summary_lines.append("Fast-IPD share")
	for threshold_ms in (1.0, 5.0, 10.0, 20.0):
		ratio = float(np.mean(ipds < (threshold_ms / 1000.0)) * 100.0)
		summary_lines.append(f"Below {threshold_ms:.0f} ms: {ratio:.2f}%")

	summary_lines.append("")
	summary_lines.append(f"Burst analysis below {burst_threshold_ms:.2f} ms")
	bursts = burst_lengths(ipds, burst_threshold_s)
	if bursts:
		burst_array = np.array(bursts, dtype=float)
		summary_lines.append(f"Burst count: {len(bursts)}")
		summary_lines.append(f"Mean burst length: {np.mean(burst_array):.2f}")
		summary_lines.append(f"Max burst length: {int(np.max(burst_array))}")
		summary_lines.append(f"Median burst length: {np.median(burst_array):.2f}")
		common_bursts = Counter(bursts).most_common(5)
		summary_lines.append("Most common burst lengths: " + ", ".join(f"{length}x{count}" for length, count in common_bursts))
	else:
		summary_lines.append("No bursts found below the selected threshold.")

	summary_lines.append("")
	summary_lines.append(f"Most frequent rounded IPDs ({rounded_decimals} decimals, ms)")
	rounded_ms = np.round(ipds_ms, rounded_decimals)
	for value, count in Counter(rounded_ms).most_common(10):
		summary_lines.append(f"{value:.{rounded_decimals}f} ms: {count}")

	return "\n".join(summary_lines)


def save_histogram(ipds: np.ndarray, output_path: Path) -> bool:
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		return False

	ipds_ms = ipds * 1000.0
	fig, ax = plt.subplots(figsize=(10, 4))
	ax.hist(ipds_ms, bins=80, color="#1f77b4", alpha=0.85, edgecolor="white")
	ax.set_title("IPD distribution")
	ax.set_xlabel("IPD (ms)")
	ax.set_ylabel("Count")
	ax.grid(alpha=0.2)
	fig.tight_layout()
	fig.savefig(output_path, dpi=160)
	plt.close(fig)
	return True


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Analyze a covert timing channel IPD CSV file.")
	parser.add_argument("--csv", required=True, help="Path to the CSV file to analyze.")
	parser.add_argument("--output-dir", default="exp", help="Directory for the saved report and histogram.")
	parser.add_argument("--burst-threshold-ms", type=float, default=20.0, help="IPD threshold used for burst analysis.")
	parser.add_argument("--rounded-decimals", type=int, default=6, help="Decimals used to group repeated IPD values.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	ipds = load_ipds(csv_path)
	report = summarize(ipds=ipds, burst_threshold_ms=args.burst_threshold_ms, rounded_decimals=args.rounded_decimals)

	report_path = output_dir / f"{csv_path.stem}_analysis.txt"
	histogram_path = output_dir / f"{csv_path.stem}_histogram.png"

	report_path.write_text(report, encoding="utf-8")
	histogram_saved = save_histogram(ipds=ipds, output_path=histogram_path)

	print(report)
	print("")
	print(f"Report saved to: {report_path}")
	if histogram_saved:
		print(f"Histogram saved to: {histogram_path}")
	else:
		print("Histogram skipped because matplotlib is not installed.")


if __name__ == "__main__":
	main()
