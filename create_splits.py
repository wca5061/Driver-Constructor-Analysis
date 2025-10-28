# create_splits.py
# Build a 70/20/10 chronological split over races (by date, then season/round fallback)
# Outputs: outputs/splits/splits.csv with columns: race_id, split ∈ {TRAIN, TEST, VAL}

import os
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs")
SPLIT_DIR = os.path.join(OUT_DIR, "splits")
Path(SPLIT_DIR).mkdir(parents=True, exist_ok=True)

races_path = os.path.join(DATA_DIR, "races.csv")
races = pd.read_csv(races_path)

# Ensure we have a sortable date
if "date" not in races.columns:
    races["date"] = pd.to_datetime(races["season_id"].astype(str) + "-01-01") \
                  + pd.to_timedelta(races["round"] * 14, unit="D")
else:
    races["date"] = pd.to_datetime(races["date"])

# Sort globally by time (then by season/round as tie-breaker)
races = races.sort_values(["date", "season_id", "round"]).reset_index(drop=True)

n = len(races)
train_n = int(np.floor(0.70 * n))
test_n  = int(np.floor(0.20 * n))
val_n   = n - train_n - test_n  # ~10%

splits = (["TRAIN"] * train_n) + (["TEST"] * test_n) + (["VAL"] * val_n)
races["split"] = splits

# Save
out = races[["race_id", "season_id", "round", "date", "split"]]
out.to_csv(os.path.join(SPLIT_DIR, "splits.csv"), index=False)

print("✅ wrote", os.path.join(SPLIT_DIR, "splits.csv"))
print(out["split"].value_counts().to_dict())