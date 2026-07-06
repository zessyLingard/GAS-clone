import pandas as pd

INPUT_FILE = "data/bignet/vpn.csv"

REMOVE_FIRST = 100000
TEST_SIZE = 400000
TRAIN_SIZE = 1530000
VAL_SIZE = 170000

df = pd.read_csv(INPUT_FILE)

# Bỏ 100k đầu
df = df.iloc[REMOVE_FIRST:].reset_index(drop=True)

# Chia dữ liệu
test_df = df.iloc[:TEST_SIZE]
train_df = df.iloc[TEST_SIZE:TEST_SIZE + TRAIN_SIZE]
val_df = df.iloc[TEST_SIZE + TRAIN_SIZE:TEST_SIZE + TRAIN_SIZE + VAL_SIZE]

# Lưu
test_df.to_csv("vpn_test.csv", index=False)
train_df.to_csv("vpn_train.csv", index=False)
val_df.to_csv("vpn_validation.csv", index=False)

print(f"Train      : {len(train_df):,}")
print(f"Validation : {len(val_df):,}")
print(f"Test       : {len(test_df):,}")