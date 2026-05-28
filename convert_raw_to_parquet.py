"""
Convert biomedrxiv_sft_raw.json to train/val parquet files for SFT training.
Uses the tools schema from the existing parquet.
"""
import json
import random
import pandas as pd

RAW_JSON = "data/biomedrxiv_sft_raw.json"
EXISTING_PARQUET = "data/biomedrxiv_sft_train.parquet"
TRAIN_OUT = "data/biomedrxiv_sft_train.parquet"
VAL_OUT = "data/biomedrxiv_sft_val.parquet"
VAL_RATIO = 0.1
SEED = 42

# Parse concatenated JSON objects
with open(RAW_JSON) as f:
    content = f.read()
data = json.loads("[" + content.replace("}\n{", "},{") + "]")
print(f"Loaded {len(data)} records")

# Get tools schema from existing parquet
existing = pd.read_parquet(EXISTING_PARQUET)
tools_schema = existing.iloc[0]["tools"]
print(f"Tools schema: {len(tools_schema)} tool(s)")

# Build records
records = [{"messages": item["messages"], "tools": tools_schema} for item in data]

# Shuffle and split
random.seed(SEED)
random.shuffle(records)
n_val = max(1, int(len(records) * VAL_RATIO))
val_records = records[:n_val]
train_records = records[n_val:]

print(f"Train: {len(train_records)}, Val: {len(val_records)}")

pd.DataFrame(train_records).to_parquet(TRAIN_OUT, index=False)
pd.DataFrame(val_records).to_parquet(VAL_OUT, index=False)
print("Done.")
