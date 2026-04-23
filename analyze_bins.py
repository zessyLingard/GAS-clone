import numpy as np
import pandas as pd

# 1. Load 100k data (đoạn baseline của bạn)
data = pd.read_csv('data/bignet/vpn_legit.csv').iloc[500000:600000, 0].values

# 2. Chia 100 Bins bằng Percentile (để đảm bảo mỗi bin có lượng data bằng nhau)
bins = np.percentile(data, np.linspace(0, 100, 101))

# 3. Tính độ rộng của từng Bin
widths = np.diff(bins)

# 4. Đếm xem có bao nhiêu Bin "đủ béo" (> 0.2s) để chống Jitter 0.1s
wide_bins_count = np.sum(widths > 0.2)
print(f"Số lượng Bin đủ rộng để nhúng tin an toàn: {wide_bins_count}/100")