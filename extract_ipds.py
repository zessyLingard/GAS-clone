import pandas as pd
import numpy as np

def analyze_fast_skype(csv_file, threshold=0.020): # Chỉ quan tâm gói < 20ms
    print(f"[*] Đang phân tích các gói SIÊU NHANH của Skype từ {csv_file}...")
    df = pd.read_csv(csv_file, header=None)
    ipds = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    
    # 1. Lọc lấy các IPD siêu nhanh (Micro-bursts)
    fast_ipds = ipds[ipds < threshold]
    fast_prob = len(fast_ipds) / len(ipds)
    print(f"[*] Tỷ lệ xuất hiện gói siêu nhanh trong Skype thật: {fast_prob*100:.2f}%")
    
    # 2. Tìm kích thước chùm (Bao nhiêu gói siêu nhanh dính nhau?)
    is_fast = ipds < threshold
    burst_sizes = []
    current_size = 0
    for b in is_fast:
        if b: current_size += 1
        elif current_size > 0:
            burst_sizes.append(current_size)
            current_size = 0
            
    if not burst_sizes:
        print("[!] File Skype của bạn không có gói nào siêu nhanh.")
        return
        
    unique_sizes, counts = np.unique(burst_sizes, return_counts=True)
    popular_sizes = [size for size, count in zip(unique_sizes, counts) if count > 2]
    
    # 3. Lấy dải giá trị Min/Max của phần siêu nhanh
    min_fast = np.min(fast_ipds)
    max_fast = np.max(fast_ipds)
    
    print("\n💡 [ĐỀ XUẤT CHO ENCODER CỦA BẠN]:")
    print(f"BURST_PROB = {fast_prob:.2f}")
    print(f"num_dummies = np.random.choice({popular_sizes})")
    print(f"dummy_ipd = np.random.uniform({min_fast:.6f}, {max_fast:.6f})")

if __name__ == "__main__":
    analyze_fast_skype("data/labnet/skype_legit_ipd.csv") # Đổi tên file nếu cần