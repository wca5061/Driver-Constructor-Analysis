import os
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic_test")
TRAIN_OUT= os.environ.get("F1_TRAIN_OUT", "outputs/f1_dynamic_train")
SPLITS   = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
PRIORS   = os.path.join(TRAIN_OUT, "priors.npz")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs"); Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# Load data
entries = pd.read_csv(os.path.join(DATA_DIR, "race_entries.csv"))
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
teams   = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
races   = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
splits  = pd.read_csv(SPLITS)

# keep classified rows for perf targets (for evaluation only)
df = (entries
      .merge(drivers[["driver_id","driver_name"]], on="driver_id", how="left")
      .merge(teams[["constructor_id","constructor_name"]], on="constructor_id", how="left")
      .merge(races[["race_id","season_id","round","track_id","weather","date"]]
             if "date" in races.columns else
             races[["race_id","season_id","round","track_id","weather"]],
             on="race_id", how="left"))

df = df.merge(splits, on="race_id", how="left")
df = df[df["split"]=="TEST"].copy()

# features
df["wet"]    = (df["weather"].fillna("").str.lower().str.contains("wet|mixed")).astype(int)
# grid_c computed within race (no leakage)
df["grid_c"] = df["grid"] - df.groupby("race_id")["grid"].transform("mean")
df["street"] = df["street"].astype(int) if "street" in df.columns else 0

# Load priors learned on TRAIN
if not os.path.exists(PRIORS):
    raise FileNotFoundError(f"Missing priors from TRAIN: {PRIORS}")
npz = np.load(PRIORS, allow_pickle=True)
drv_mu_map = {d: m for d,m in zip(npz["drv_ids"], npz["drv_mus"])}
tm_mu_map  = {t: m for t,m in zip(npz["tm_ids"],  npz["tm_mus"])}
alpha0     = float(npz.get("alpha_mean_global", 0.0))
beta0      = npz.get("beta_mean_global", np.zeros(3))

# make sure beta aligns to [grid_c, street, wet]
if beta0.shape[0] != 3:
    beta0 = np.zeros(3)

# Predicted perf = alpha0 + driver_prior_mean + team_prior_mean + X @ beta0
X = df[["grid_c","street","wet"]].to_numpy(float)
drv = df["driver_id"].map(drv_mu_map).fillna(0.0).to_numpy(float)
tm  = df["constructor_id"].map(tm_mu_map).fillna(0.0).to_numpy(float)
df["pred_perf"] = alpha0 + drv + tm + (X @ beta0)

# For reporting: build same columns as f1_dynamic_update outputs
keep = ["race_id","season_id","round","driver_id","driver_name","constructor_id",
        "constructor_name","grid","finish_position","street","wet","pred_perf"]
out = df[keep].sort_values(["season_id","round","driver_name"])
out_path = os.path.join(OUT_DIR, "dcsi_race_test_pred.csv")
out.to_csv(out_path, index=False)
print("✅ Wrote TEST predictions (no fitting) to:", out_path)
