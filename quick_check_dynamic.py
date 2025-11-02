# quick_check_dynamic.py
# Lightweight sanity checks on dcsi_race.csv for a single split (TRAIN/TEST/VAL)
# Prints a few top tables and basic stats.

import os
from pathlib import Path
import pandas as pd
import numpy as np

OUT_DIR     = os.environ.get("F1_OUT_DIR", "outputs/f1_dynamic_test")   # point to split folder
SPLITS_CSV  = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
SPLIT_ENV   = os.environ.get("F1_SPLIT_TARGET", "").upper()

def infer_split_from_outdir(path: str):
    l = path.lower()
    if "train" in l: return "TRAIN"
    if "test"  in l: return "TEST"
    if "val"   in l: return "VAL"
    return ""

SPLIT_TARGET = SPLIT_ENV or infer_split_from_outdir(OUT_DIR) or "UNSPECIFIED"

race_path = os.path.join(OUT_DIR, "dcsi_race.csv")
if not os.path.exists(race_path):
    raise FileNotFoundError(f"Missing {race_path}. Run f1_dynamic_update.py or f1_parent_model.py first.")

r = pd.read_csv(race_path)

# Enforce split filtering if we have split metadata
if os.path.exists(SPLITS_CSV) and "race_id" in r.columns:
    s = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
    r = r.merge(s, on="race_id", how="left")
    if SPLIT_TARGET in {"TRAIN","TEST","VAL"}:
        r = r[r["split"] == SPLIT_TARGET]
    else:
        r = r[r["split"].isin(["TRAIN","TEST"])]  # default safeguard

if r.empty:
    raise RuntimeError(f"No rows after filtering for split={SPLIT_TARGET}. Check inputs.")

# Basic shapes
n_races = r["race_id"].nunique()
n_starts = len(r)
n_drivers = r["driver_id"].nunique()
n_teams = r["constructor_id"].nunique()

print(f"[{SPLIT_TARGET}] races={n_races}, starts={n_starts}, drivers={n_drivers}, teams={n_teams}")

# Top "outperforming car" drivers
if {"driver_eff_mean","team_eff_mean"}.issubset(r.columns):
    agg = (r.groupby(["driver_id","driver_name"], as_index=False)
             .agg(driver_eff=("driver_eff_mean","mean"),
                  team_eff=("team_eff_mean","mean"),
                  starts=("race_id","nunique")))
    agg["outperforming_car"] = agg["driver_eff"] - agg["team_eff"]
    top = agg.sort_values("outperforming_car", ascending=False).head(10)
    print("\nTop 10—Drivers outperforming car:")
    print(top[["driver_name","outperforming_car","driver_eff","team_eff","starts"]])

# Team strength
if "team_eff_mean" in r.columns:
    tt = (r.groupby(["constructor_id","constructor_name"], as_index=False)
            .agg(team_strength=("team_eff_mean","mean"),
                 starts=("race_id","nunique"))
          ).sort_values("team_strength", ascending=False).head(10)
    print("\nTop 10—Constructor strength:")
    print(tt[["constructor_name","team_strength","starts"]])

# Calibration proxy: predicted rank vs finish (if available)
if {"pred_perf","finish_position"}.issubset(r.columns):
    cal = r[["race_id","driver_id","pred_perf","finish_position"]].copy()
    cal["pred_rank"] = cal.groupby("race_id")["pred_perf"].rank(ascending=False, method="average")
    corr = cal[["pred_rank","finish_position"]].corr().iloc[0,1]
    print(f"\nCalibration proxy (Spearman-ish via Pearson on ranks): {corr:.3f} "
          "(lower is better since finish_position is smaller for better results)")

print("\nDone.")
