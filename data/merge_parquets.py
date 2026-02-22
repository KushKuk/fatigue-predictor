import pandas as pd
import glob

files = glob.glob("data/parquets/*.parquet")
print(f"Found {len(files)} parquet files")

dfs = [pd.read_parquet(f) for f in files]

merged = pd.concat(dfs, ignore_index=True)

merged.to_parquet("data/merged.parquet", index=False)

print("Merged dataset saved to data/merged.parquet")
print("Final shape:", merged.shape)