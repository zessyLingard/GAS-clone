import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILE_PATH = 'data/labnet/timestamps.csv'


def load_ipd_series(csv_file):
    """Load IPD data robustly from common column formats."""
    df = pd.read_csv(csv_file)
    cols_lower = {c.lower(): c for c in df.columns}

    # Common IPD column names
    for candidate in ['ipds', 'ipd']:
        if candidate in cols_lower:
            series = pd.to_numeric(df[cols_lower[candidate]], errors='coerce').dropna()
            return series.values

    # If the file stores timestamps, convert to IPD by first difference
    for candidate in ['timestamps', 'timestamp', 'time', 'ts']:
        if candidate in cols_lower:
            ts = pd.to_numeric(df[cols_lower[candidate]], errors='coerce').dropna().values
            return np.diff(ts)

    # Fallback for one-column files
    if df.shape[1] == 1:
        series = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        return series.values

    raise ValueError(f"Không xác định được cột IPD/Timestamp trong file: {csv_file}")

def deep_stats_analysis(csv_file):
    # 1. Load dữ liệu và làm sạch
    data = load_ipd_series(csv_file)
    data = data[data > 0] # Loại bỏ giá trị 0 tuyệt đối

    if data.size == 0:
        print("Không có dữ liệu IPD hợp lệ sau khi làm sạch.")
        return

    print("="*60)
    print("📊 THỐNG KÊ CHI TIẾT TOÀN BỘ LUỒNG TRAFFIC (ALL-IN)")
    print("="*60)

    # 2. Thống kê mô tả
    stats = {
        "Tổng số gói": len(data),
        "Trung bình (Mean)": np.mean(data),
        "Trung vị (Median)": np.median(data),
        "Độ lệch chuẩn (Std)": np.std(data),
        "Nhỏ nhất (Min)": np.min(data),
        "Lớn nhất (Max)": np.max(data),
        "Phân vị 75th (P75)": np.percentile(data, 75),
        "Phân vị 90th (P90)": np.percentile(data, 90),
        "Phân vị 95th (P95)": np.percentile(data, 95)
    }

    for k, v in stats.items():
        if k == "Tổng số gói":
            print(f"{k:25}: {int(v):,}")
        else:
            print(f"{k:25}: {v:.6f} giây")

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(12, 6))

    # Biểu đồ 1: Histogram với Log-scale (Bắt buộc phải dùng log-scale)
    plt.subplot(1, 2, 1)
    sns.histplot(data, bins=100, log_scale=True, kde=True, color='royalblue')
    plt.axvline(0.1, color='red', linestyle='--', label='Ngưỡng Idle (0.1s)')
    plt.title("Phân phối IPD (Log-scale)")
    plt.xlabel("Thời gian (giây) - Log Scale")
    plt.ylabel("Tần suất")
    plt.legend()

    # Biểu đồ 2: Cumulative Distribution Function (CDF)
    plt.subplot(1, 2, 2)
    sns.ecdfplot(data, color='forestgreen')
    plt.axvline(0.1, color='red', linestyle='--')
    plt.title("Hàm phân phối tích lũy (CDF)")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Tỷ lệ tích lũy")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

deep_stats_analysis(FILE_PATH)