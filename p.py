import pandas as pd

# đọc file, giữ nguyên dạng string để không mất precision
df = pd.read_csv("data/labnet/legit.csv", dtype={"Time": str})

# lấy đúng cột Time
time_col = df["Time"].iloc[:610000]

# đổi tên
time_col = time_col.rename("IPDs")

# lưu ra file mới (1 cột duy nhất)
time_col.to_csv("ipd_only.csv", index=False)