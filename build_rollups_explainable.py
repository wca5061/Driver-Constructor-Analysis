# build_rollups_explainable.py
# Explainable roll-ups for a single split (TRAIN/TEST/VAL):
#  - "Outperforming the car" = driver_eff_mean - team_eff_mean
#  - Driver and team leaderboards (overall + per-season)
#  - Condition splits (street vs permanent, wet vs dry)
#  - Figures saved to figs/
#
# Env:
#   F1_OUT_DIR     (default: outputs/f1_dynamic)   # point at split folder (train/test/val)
#   F1_SPLITS_CSV  (default: outputs/splits/splits.csv)
#   F1_SPLIT_TARGET TRAIN|TEST|VAL (optional; inferred from OUT_DIR if unset)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------- Paths & split handling ---------------
OUT_DIR = os.environ.get("F1_OUT_DIR", "outputs/f1_dynamic")
FIG_DIR = os.path.join(OUT_DIR, "figs")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

SPLITS_CSV  = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
SPLIT_ENV   = os.environ.get("F1_SPLIT_TARGET", "").upper()

def infer_split_from_outdir(path: str):
    lower = path.lower()
    if "train" in lower: return "TRAIN"
    if "test"  in lower: return "TEST"
    if "val"   in lower: return "VAL"
    return ""

SPLIT_TARGET = SPLIT_ENV or infer_split_from_outdir(OUT_DIR) or "UNSPECIFIED"

RACE_PATH = os.path.join(OUT_DIR, "dcsi_race.csv")
PROB_PATH = os.path.join(OUT_DIR, "dcsi_probs.csv")  # optional
CUM_DRV   = os.path.join(OUT_DIR, "dcsi_cumulative_drivers.csv")  # optional for context
CUM_TM    = os.path.join(OUT_DIR, "dcsi_cumulative_constructors.csv")  # optional

if not os.path.exists(RACE_PATH):
    raise FileNotFoundError(f"Missing {RACE_PATH}. Run f1_dynamic_update.py for this split first.")

r = pd.read_csv(RACE_PATH)

# If splits.csv exists and race_id is present, enforce split filter.
if os.path.exists(SPLITS_CSV) and "race_id" in r.columns:
    s = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
    r = r.merge(s, on="race_id", how="left")
    if SPLIT_TARGET in {"TRAIN","TEST","VAL"}:
        r = r[r["split"] == SPLIT_TARGET]
    else:
        # default safeguard: exclude VAL unless explicitly chosen
        r = r[r["split"].isin(["TRAIN","TEST"])]

# --------------- Required columns ---------------
req = ["race_id","driver_id","driver_name","constructor_id","constructor_name",
       "season_id","round","pred_perf","driver_eff_mean","team_eff_mean"]
for c in req:
    if c not in r.columns:
        raise KeyError(f"Required column missing from dcsi_race.csv: {c}")

# Some visuals/rollups use these if available; create safe fallbacks
if "street" not in r.columns:
    r["street"] = 0
if "wet" not in r.columns:
    r["wet"] = 0

# --------------- Core derived metrics ---------------
r["delta"] = r["driver_eff_mean"] - r["team_eff_mean"]  # + = driver > car
if "dcsi_driver_share_entry" in r.columns:
    r["driver_share"] = r["dcsi_driver_share_entry"]
else:
    r["driver_share"] = np.abs(r["driver_eff_mean"]) / (
        np.abs(r["driver_eff_mean"]) + np.abs(r["team_eff_mean"]) + 1e-9
    )
r["team_share"] = 1.0 - r["driver_share"]

# --------------- Driver roll-ups ---------------
driver_overall = (r.groupby(["driver_id","driver_name"], as_index=False)
                    .agg(outperforming_car=("delta","mean"),
                         driver_share_mean=("driver_share","mean"),
                         team_share_mean=("team_share","mean"),
                         avg_pred_perf=("pred_perf","mean"),
                         starts=("race_id","nunique")))
driver_overall.sort_values("outperforming_car", ascending=False, inplace=True)
driver_overall.to_csv(os.path.join(OUT_DIR, "rollup_drivers_overall.csv"), index=False)

driver_season = (r.groupby(["season_id","driver_id","driver_name"], as_index=False)
                   .agg(outperforming_car=("delta","mean"),
                        driver_share_mean=("driver_share","mean"),
                        avg_pred_perf=("pred_perf","mean"),
                        starts=("race_id","nunique")))
driver_season.sort_values(["season_id","outperforming_car"], ascending=[True,False], inplace=True)
driver_season.to_csv(os.path.join(OUT_DIR, "rollup_drivers_by_season.csv"), index=False)

# Condition splits
cond_rows = []
for key, g in r.groupby(["driver_id","driver_name","street","wet"], observed=True):
    d_id, d_name, street, wet = key
    cond_rows.append({
        "driver_id": d_id,
        "driver_name": d_name,
        "street": int(street),
        "wet": int(wet),
        "outperforming_car": g["delta"].mean(),
        "driver_share_mean": g["driver_share"].mean(),
        "starts": g["race_id"].nunique()
    })
driver_cond = pd.DataFrame(cond_rows)
driver_cond.to_csv(os.path.join(OUT_DIR, "rollup_drivers_condition_split.csv"), index=False)

# --------------- Team roll-ups ---------------
team_overall = (r.groupby(["constructor_id","constructor_name"], as_index=False)
                  .agg(team_strength=("team_eff_mean","mean"),
                       avg_pred_perf=("pred_perf","mean"),
                       starts=("race_id","nunique")))
team_overall.sort_values("team_strength", ascending=False, inplace=True)
team_overall.to_csv(os.path.join(OUT_DIR, "rollup_teams_overall.csv"), index=False)

team_season = (r.groupby(["season_id","constructor_id","constructor_name"], as_index=False)
                 .agg(team_strength=("team_eff_mean","mean"),
                      avg_pred_perf=("pred_perf","mean"),
                      starts=("race_id","nunique")))
team_season.sort_values(["season_id","team_strength"], ascending=[True,False], inplace=True)
team_season.to_csv(os.path.join(OUT_DIR, "rollup_teams_by_season.csv"), index=False)

# --------------- Optional: integrate average p(win) if probs exist ---------------
if os.path.exists(PROB_PATH):
    probs = pd.read_csv(PROB_PATH)
    if set(["race_id","driver_id","p_win"]).issubset(probs.columns):
        pw = (probs.groupby(["driver_id"], as_index=False)["p_win"].mean()
                    .rename(columns={"p_win":"p_win_mean"}))
        driver_overall = driver_overall.merge(pw, on="driver_id", how="left")
        driver_overall.to_csv(os.path.join(OUT_DIR, "rollup_drivers_overall.csv"), index=False)

# --------------- Figures ---------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

topN = 20

# Top drivers punching above car
top = driver_overall.head(topN).iloc[::-1]
plt.figure(figsize=(8,7))
plt.barh(top["driver_name"], top["outperforming_car"])
plt.xlabel("Driver Effect – Team Effect (mean)")
plt.title(f"{SPLIT_TARGET}: Top {topN} Drivers Outperforming Their Cars")
savefig(os.path.join(FIG_DIR, "top_outperformers_drivers.png"))

# Top "car carrying driver" (more negative delta)
bot = driver_overall.sort_values("outperforming_car").head(topN).iloc[::-1]
plt.figure(figsize=(8,7))
plt.barh(bot["driver_name"], bot["outperforming_car"])
plt.xlabel("Driver Effect – Team Effect (mean)")
plt.title(f"{SPLIT_TARGET}: Top {topN} 'Car Carrying Driver' Cases")
savefig(os.path.join(FIG_DIR, "car_carrying_drivers.png"))

# Team strength
tt = team_overall.head(15).iloc[::-1]
plt.figure(figsize=(8,7))
plt.barh(tt["constructor_name"], tt["team_strength"])
plt.axvline(0, linewidth=1)
plt.xlabel("Constructor Effect (mean)")
plt.title(f"{SPLIT_TARGET}: Top Constructors (Strength)")
savefig(os.path.join(FIG_DIR, "teams_strength_bar.png"))

# Driver vs Team effect scatter (overall means per driver)
drv_scatter = (r.groupby(["driver_id","driver_name"], as_index=False)
                 .agg(driver_eff=("driver_eff_mean","mean"),
                      team_eff=("team_eff_mean","mean")))
plt.figure(figsize=(7,7))
plt.scatter(drv_scatter["team_eff"], drv_scatter["driver_eff"], s=30, alpha=0.8)
for _, row in drv_scatter.sort_values("driver_eff", ascending=False).head(12).iterrows():
    # label by last name (or last token)
    plt.text(row["team_eff"], row["driver_eff"], row["driver_name"].split()[-1], fontsize=8, va="bottom")
plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
plt.xlabel("Constructor Effect (mean)"); plt.ylabel("Driver Effect (mean)")
plt.title(f"{SPLIT_TARGET}: Driver vs Constructor Effects")
savefig(os.path.join(FIG_DIR, "driver_vs_constructor_scatter_named.png"))

print("✅ Wrote split roll-ups to:", OUT_DIR)
print("   - rollup_drivers_overall.csv")
print("   - rollup_drivers_by_season.csv")
print("   - rollup_drivers_condition_split.csv")
print("   - rollup_teams_overall.csv")
print("   - rollup_teams_by_season.csv")
print("🎨 Plots saved to:", FIG_DIR)
