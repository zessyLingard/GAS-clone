import os
import pandas as pd
import numpy as np

print("=" * 60)
print("PHÂN TÍCH ĐỘNG HỌC FILE DO-H THỰC TẾ")
print("=" * 60)

# 1. Tải file của bạn
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'labnet', 'real_ipds_doH.csv')
try:
    df = pd.read_csv(file_path)
    real_ipds = df['IPDs'].values
except FileNotFoundError:
    print(f"Không tìm thấy file tại {file_path}. Vui lòng kiểm tra lại đường dẫn.")
    real_ipds = np.array([])

if len(real_ipds) > 0:
    # 2. Thống kê cơ bản
    print(f"Tổng số gói tin (IPDs): {len(real_ipds):,}")
    print(f"Trễ trung bình (Mean):  {np.mean(real_ipds):.6f} giây")
    print(f"Trễ lớn nhất (Max):     {np.max(real_ipds):.6f} giây")
    print("-" * 60)

    # 3. Phân rã theo Percentile (Để tìm vùng Micro-burst)
    percentiles = [50, 75, 80, 90, 95, 98, 99, 99.5, 99.9, 100]
    print("BẢNG PERCENTILE (Bao nhiêu % gói tin có độ trễ nhỏ hơn mức này?):")
    for p in percentiles:
        val = np.percentile(real_ipds, p)
        print(f" - {p:5.1f}% gói tin trễ <= {val:.6f} giây")
    
    print("-" * 60)
    
    # 4. Tìm vị trí % cho các mốc Vật lý mục tiêu (Để tìm vùng Macro-delay)
    targets = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    print("BẢNG MỐC VẬT LÝ (Thời gian này nằm ở khúc nào của đồ thị?):")
    for t in targets:
        prob = np.mean(real_ipds <= t) * 100
        print(f" - Mốc {t:.3f} giây tương đương vị trí {prob:.4f}% trên CDF")