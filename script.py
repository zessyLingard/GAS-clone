import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("[1] Đang đọc dữ liệu thời gian...")
# Đọc file timestamp, bỏ qua các dòng rỗng nếu có
with open('data/labnet/mss_ipd.csv', 'r') as f:
    times = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            times.append(float(line))
        except ValueError:
            print(f"Bỏ qua dòng không hợp lệ: {line}")

print(f"    -> Tổng số gói tin bắt được: {len(times)}")

# Tính khoảng cách giữa gói sau và gói trước
times = np.array(times)
ipds = np.diff(times)

print("[2] Đang dọn dẹp dữ liệu (Lọc rác vật lý)...")
# Giữ lại các IPD > 0 (bỏ các gói tin đến cùng 1 lúc do lag) 
# VÀ giữ các IPD < 5.0 giây (khoảng lặng > 5s là do bạn đi vệ sinh chứ không phải macro-delay tự nhiên)
clean_ipds = ipds[(ipds > 0.000001) & (ipds < 5.0)]

print(f"    -> Số lượng IPD hợp lệ để đưa vào GAN: {len(clean_ipds)}")

# Lưu ra file CSV cho TimeGAN
df = pd.DataFrame(clean_ipds, columns=['IPD'])
df.to_csv('real_ipds_doH.csv', index=False)
print("[3] Đã lưu thành công ra file: real_ipds_doH.csv")

# ==========================================
# PHẦN ĂN ĐIỂM BÁO CÁO: VẼ ĐỒ THỊ CDF
# ==========================================
print("[4] Đang vẽ đồ thị CDF...")
sorted_ipds = np.sort(clean_ipds)
# Tính xác suất tích lũy (từ 0 đến 1)
yvals = np.arange(len(sorted_ipds)) / float(len(sorted_ipds) - 1)

plt.figure(figsize=(10, 6))
plt.plot(sorted_ipds, yvals, color='red', linewidth=2)

plt.title("Đường cong Phân phối Tích lũy (CDF) của luồng DoH thực tế")
plt.xlabel("Thời gian IPD (giây) - Trục X")
plt.ylabel("Tỷ lệ % gói tin (Xác suất CDF) - Trục Y")
plt.xscale('log') # Dùng thang Logarit để nhìn rõ cả Micro-burst và Macro-delay
plt.grid(True, which="both", ls="--")

plt.savefig('doh_cdf_real.png', dpi=300)
print("    -> Đã lưu biểu đồ thành doh_cdf_real.png. Hãy mang ảnh này đi báo cáo!")