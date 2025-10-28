# build_rollups_explainable.py
# Create explainable roll-ups:
#  - "Punching above the car" (driver_eff - team_eff)
#  - "Car carrying driver" (negative side)
#  - Team strength leaderboards
#  - Condition splits: street/permanent, wet/dry
#  - Plots saved to figs/
#
# Inputs:
#   outputs/f1_dynamic/dcsi_race.csv
#   outputs/f1_dynamic/dcsi_probs.csv (optional; blends for summaries)
#   outputs/f1_dynamic/dcsi_cumulative_{drivers,constructors}.csv (optional)
#
# Env:
#   F1_OUT_DIR (default: outputs/f1_dynamic)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = os.environ.get("F1_OUT_DIR", "outputs/f1_dynamic")
FIG_DIR = os.path.join(OUT_DIR, "figs")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

RACE_PATH = os.path.join(OUT_DIR, "dcsi_race.csv")
PROB_PATH = os.path.join(OUT_DIR, "dcsi_probs.csv")
CUM_DRV   = os.path.join(OUT_DIR, "dcsi_cumulative_drivers.csv")
CUM_TM    = os.path.join(OUT_DIR, "dcsi_cumulative_constructors.csv")

if not os.path.exists(RACE_PATH):
    raise FileNotFoundError(f"Missing {RACE_PATH}. Run f1_dynamic_update.py first.")

r = pd.read_csv(RACE_PATH)

# ensure required fields exist
req = ["race_id","driver_id","driver_name","constructor_id","constructor_name",
       "season_id","round","finish_position","pred_perf",
       "driver_eff_mean","team_eff_mean","dcsi_weight_driver","dcsi_weight_team","street","wet"]
for c in req:
    if c not in r.columns:
        raise KeyError(f"Required column missing from dcsi_race.csv: {c}")

# Optional probabilities
probs = pd.read_csv(PROB_PATH) if os.path.exists(PROB_PATH) else None

# ------------- Core derived metrics -------------
r["delta"] = r["driver_eff_mean"] - r["team_eff_mean"]  # + = driver > car; - = car carrying
r["driver_share"] = r["dcsi_driver_share_entry"] if "dcsi_driver_share_entry" in r.columns else \
                    (np.abs(r["driver_eff_mean"]) / (np.abs(r["driver_eff_mean"]) + np.abs(r["team_eff_mean"]) + 1e-9))
r["team_share"]   = 1.0 - r["driver_share"]

# ------------- Overall driver roll-ups -------------
driver_overall = (r.groupby(["driver_id","driver_name"], as_index=False)
                    .agg(outperforming_car=("delta","mean"),
                         driver_share_mean=("driver_share","mean"),
                         team_share_mean=("team_share","mean"),
                         avg_pred_perf=("pred_perf","mean"),
                         starts=("race_id","nunique")))
driver_overall.sort_values("outperforming_car", ascending=False, inplace=True)
driver_overall.to_csv(os.path.join(OUT_DIR, "rollup_drivers_overall.csv"), index=False)

# ------------- Per-season driver roll-ups -------------
driver_season = (r.groupby(["season_id","driver_id","driver_name"], as_index=False)
                   .agg(outperforming_car=("delta","mean"),
                        driver_share_mean=("driver_share","mean"),
                        avg_pred_perf=("pred_perf","mean"),
                        starts=("race_id","nunique")))
driver_season.sort_values(["season_id","outperforming_car"], ascending=[True,False], inplace=True)
driver_season.to_csv(os.path.join(OUT_DIR, "rollup_drivers_by_season.csv"), index=False)

# ------------- Condition splits (street/permanent, wet/dry) -------------
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

# ------------- Team strength roll-ups -------------
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

# ------------- Optional blend: bring in p_win for a quick explainable table -------------
if probs is not None and set(["race_id","driver_id","p_win"]).issubset(probs.columns):
    pw = (probs.groupby(["driver_id"], as_index=False)["p_win"].mean()
                 .rename(columns={"p_win":"p_win_mean"}))
    driver_overall = driver_overall.merge(pw, on="driver_id", how="left")
    driver_overall.to_csv(os.path.join(OUT_DIR, "rollup_drivers_overall.csv"), index=False)

# ------------- Plots: Top Outperformers & Team Strength -------------
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
plt.title(f"Top {topN} Drivers Outperforming Their Cars")
savefig(os.path.join(FIG_DIR, "top_outperformers_drivers.png"))

# Top "car carrying driver" (negative delta)
bot = driver_overall.tail(topN).sort_values("outperforming_car").iloc[:topN][::-1]
plt.figure(figsize=(8,7))
plt.barh(bot["driver_name"], bot["outperforming_car"])
plt.xlabel("Driver Effect – Team Effect (mean)")
plt.title(f"Top {topN} 'Car Carrying Driver' Cases (More Negative)")
savefig(os.path.join(FIG_DIR, "car_carrying_drivers.png"))

# Team strength
tt = team_overall.head(15).iloc[::-1]
plt.figure(figsize=(8,7))
plt.barh(tt["constructor_name"], tt["team_strength"])
plt.axvline(0, linewidth=1)
plt.xlabel("Constructor Effect (mean)")
plt.title("Top Constructors (Strength)")
savefig(os.path.join(FIG_DIR, "teams_strength_bar.png"))

# Driver vs Team effect scatter (overall means per driver)
drv_scatter = (r.groupby(["driver_id","driver_name"], as_index=False)
                 .agg(driver_eff=("driver_eff_mean","mean"),
                      team_eff=("team_eff_mean","mean")))
plt.figure(figsize=(7,7))
plt.scatter(drv_scatter["team_eff"], drv_scatter["driver_eff"], s=30, alpha=0.8)
# label a few notable points
for _, row in drv_scatter.sort_values("driver_eff", ascending=False).head(12).iterrows():
    plt.text(row["team_eff"], row["driver_eff"], row["driver_name"].split()[-1], fontsize=8, va="bottom")
plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
plt.xlabel("Constructor Effect (mean)"); plt.ylabel("Driver Effect (mean)")
plt.title("Driver vs Constructor Effects")
savefig(os.path.join(FIG_DIR, "driver_vs_constructor_scatter_named.png"))

print("✅ Wrote roll-ups to:", OUT_DIR)
print("   - rollup_drivers_overall.csv")
print("   - rollup_drivers_by_season.csv")
print("   - rollup_drivers_condition_split.csv")
print("   - rollup_teams_overall.csv")
print("   - rollup_teams_by_season.csv")
print("🎨 Plots saved to:", FIG_DIR)
