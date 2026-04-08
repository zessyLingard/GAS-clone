import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_vpn_csv(csv_file, burst_threshold=0.1):
    """
    Phân tích file CSV chứa IPD để tìm ra quy luật cấu trúc Burst.
    """
    print(f"[*] Đang đọc file dữ liệu {csv_file}...")
    
    try:
        # Đọc file CSV, giả định IPD nằm ở cột đầu tiên
        df = pd.read_csv(csv_file, header=None)
        ipds = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    except Exception as e:
        print(f"[!] Lỗi khi đọc file CSV: {e}")
        return

    print(f"[*] Tổng số IPD đọc được: {len(ipds)}")

    # 1. Phân tích Burst
    is_burst = ipds < burst_threshold
    burst_probability = np.mean(is_burst)
    
    # Tính kích thước (số lượng gói liên tiếp) trong từng đợt Burst
    burst_sizes = []
    current_burst_size = 0
    
    for b in is_burst:
        if b:
            current_burst_size += 1
        else:
            if current_burst_size > 0:
                burst_sizes.append(current_burst_size)
                current_burst_size = 0
                
    if current_burst_size > 0:
        burst_sizes.append(current_burst_size)

    # 2. Thống kê chi tiết
    if not burst_sizes:
        print(f"[!] Không tìm thấy đợt burst nào dưới {burst_threshold}s.")
        return

    unique_sizes, counts = np.unique(burst_sizes, return_counts=True)
    
    print("\n" + "="*50)
    print(" 🎯 THỐNG KÊ CHI TIẾT CẤU TRÚC BURST 🎯")
    print("="*50)
    print(f"Tỷ lệ gói Burst tổng thể: {burst_probability*100:.2f}%")
    print("-" * 50)
    print(" Kích thước chùm (gói) | Số lần xuất hiện")
    print("-" * 50)
    for size, count in zip(unique_sizes, counts):
        print(f" {size:20d} | {count:15d}")
    print("="*50)

    # Tự động trích xuất các kích thước phổ biến nhất (xuất hiện > 2 lần) để tạo mảng choice
    popular_sizes = [size for size, count in zip(unique_sizes, counts) if count > 2]
    
    # Nếu ít quá thì lấy hết
    if not popular_sizes:
        popular_sizes = unique_sizes.tolist()
        
    print(f"\n💡 [ĐỀ XUẤT CHO ENCODER]:")
    print(f"Bạn hãy copy mảng sau thay vào np.random.choice() trong encoder_noise.py:")
    print(f"num_dummies = np.random.choice({popular_sizes})")
    
    # 3. Vẽ biểu đồ phân phối kích thước Burst
    print("\n[*] Đang vẽ biểu đồ cấu trúc chùm Burst (Đóng cửa sổ để kết thúc)...")
    plt.figure(figsize=(10, 6))
    
    # Bỏ qua các chùm siêu siêu dài (outliers > 100 gói) để biểu đồ dễ nhìn nếu có
    plot_data = [s for s in burst_sizes if s < 100]
    
    plt.hist(plot_data, bins=range(1, max(plot_data)+2), align='left', color='#E24A4A', edgecolor='black', alpha=0.8)
    plt.title('Phân phối kích thước chùm Burst - Mạng VPN thực tế', fontsize=14)
    plt.xlabel('Số lượng gói tin liên tiếp trong 1 chùm (Gói)', fontsize=12)
    plt.ylabel('Tần suất xuất hiện', fontsize=12)
    plt.xticks(range(1, min(max(plot_data)+1, 30))) # Đánh dấu trục X cho dễ nhìn
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ĐỔI TÊN FILE CSV CỦA BẠN VÀO ĐÂY (File gốc chưa lọc nhé)
    TARGET_CSV = "data/bignet/vpn.csv" 
    
    if os.path.exists(TARGET_CSV):
        analyze_vpn_csv(TARGET_CSV, burst_threshold=0.1)
    else:
        print(f"[!] Không tìm thấy file {TARGET_CSV}. Hãy đảm bảo đúng đường dẫn.")