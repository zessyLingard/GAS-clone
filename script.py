import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE CSV CỦA BẠN
# ==========================================
REAL_FILE = "data/labnet/real_ipds_doH.csv"      # File gốc
COVERT_FILE = "data/labnet/vpn_noise_v3.csv"       # File output (đổi tên lại cho khớp với file của bạn)

def load_data(file_path):
    print(f"Loading {file_path}...")
    # Đọc file, xử lý lỗi parse và drop NA
    df = pd.read_csv(file_path, header=None)
    data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    return data

def plot_cdf(real_data, covert_data):
    print("Vẽ biểu đồ CDF...")
    plt.figure(figsize=(10, 6))
    
    # Tính toán CDF cho Real
    sorted_real = np.sort(real_data)
    y_real = np.arange(1, len(sorted_real) + 1) / len(sorted_real)
    plt.plot(sorted_real, y_real, label='Real DoH Traffic', color='blue', linewidth=2)
    
    # Tính toán CDF cho Covert
    sorted_covert = np.sort(covert_data)
    y_covert = np.arange(1, len(sorted_covert) + 1) / len(sorted_covert)
    plt.plot(sorted_covert, y_covert, label='Covert Traffic', color='red', linestyle='--', linewidth=2)
    
    plt.xlim(0, 1.0) # Zoom vào dải từ 0 đến 1 giây để nhìn rõ vùng PAYLOAD_THRESHOLD (0.10s)
    plt.title('Cumulative Distribution Function (CDF) Comparison', fontsize=14)
    plt.xlabel('Inter-Packet Delay (seconds)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.axvline(x=0.10, color='gray', linestyle=':', label='Payload Threshold (0.10s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('cdf_comparison.png', dpi=300)
    print("Đã lưu cdf_comparison.png")
    plt.close()

def plot_autocorrelation(real_data, covert_data, lags=50):
    print("Vẽ biểu đồ Autocorrelation (ACF)...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # Lấy một sample đủ lớn (ví dụ 10,000 IPDs) để tính ACF tránh tràn RAM
    sample_size = min(10000, len(real_data), len(covert_data))
    
    # ACF cho Real
    plot_acf(real_data[:sample_size], lags=lags, ax=axes[0], color='blue', title='Autocorrelation: Real DoH Traffic')
    
    # ACF cho Covert
    plot_acf(covert_data[:sample_size], lags=lags, ax=axes[1], color='red', title='Autocorrelation: Covert Traffic')
    
    plt.xlabel('Lag (Số gói tin trễ)', fontsize=12)
    plt.tight_layout()
    plt.savefig('acf_comparison.png', dpi=300)
    print("Đã lưu acf_comparison.png")
    plt.close()

def main():
    real_data = load_data(REAL_FILE)
    covert_data = load_data(COVERT_FILE)
    
    print(f"Số lượng IPD - Real: {len(real_data)}, Covert: {len(covert_data)}")
    
    plot_cdf(real_data, covert_data)
    plot_autocorrelation(real_data, covert_data)
    print("Hoàn tất! Hãy mở 2 ảnh png lên để phân tích.")

if __name__ == "__main__":
    main()