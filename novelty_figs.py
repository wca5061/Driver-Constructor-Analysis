# novelty_figs.py
#
# Create static figures that showcase our novel dynamic model:
#  - Driver DCSI time series
#  - Constructor DCSI time series
#  - Top "outperforming car" drivers
#
# Run from repo root:
#   python novelty_figs.py

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------- Config ---------
TRAIN_DIR = Path("outputs/f1_dynamic_train_pit_sc")
ROLLUP_DIR = Path("outputs/f1_dynamic_test_pit_sc")  # rollups usually built on TEST

dcsi_race_path = TRAIN_DIR / "dcsi_race.csv"
drivers_rollup_path = ROLLUP_DIR / "rollup_drivers_overall.csv"

OUT_DIR = Path("outputs/figs_novelty")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Using TRAIN dcsi_race: {dcsi_race_path}")
print(f"[INFO] Using TEST rollup_drivers_overall: {drivers_rollup_path}")
print(f"[INFO] Saving figures to: {OUT_DIR}")

# --------- Load base data ---------
if not dcsi_race_path.exists():
    raise FileNotFoundError(f"Missing {dcsi_race_path}. Run f1_dynamic_update.py with pit+SC first.")

race_df = pd.read_csv(dcsi_race_path)

if drivers_rollup_path.exists():
    drv_rollup = pd.read_csv(drivers_rollup_path)
else:
    drv_rollup = None

# --------- Helper to choose a "hero" driver/constructor ---------
# pick the driver with most starts in TRAIN
driver_counts = race_df["driver_name"].value_counts()
driver_name = driver_counts.index[0]
print(f"[INFO] Using driver for time series: {driver_name}")

# pick the constructor with most starts in TRAIN
team_counts = race_df["constructor_name"].value_counts()
team_name = team_counts.index[0]
print(f"[INFO] Using constructor for time series: {team_name}")

# --------- Figure 1: Driver DCSI time series ---------
driver_df = race_df[race_df["driver_name"] == driver_name].copy()
driver_df = driver_df.sort_values(["season_id", "round"])

plt.figure(figsize=(7, 4))
for season, g in driver_df.groupby("season_id"):
    plt.plot(
        g["round"],
        g["driver_eff_mean"],
        marker="o",
        label=str(season)
    )

plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
plt.xlabel("Round")
plt.ylabel("Driver Effect (DCSI component)")
plt.title(f"Dynamic Driver Effect Over Time – {driver_name}")
plt.legend(title="Season", fontsize=8)
plt.tight_layout()

out_driver_ts = OUT_DIR / "fig_driver_dcsi_timeseries.png"
plt.savefig(out_driver_ts, dpi=200)
plt.close()
print(f"[OK] Wrote {out_driver_ts}")

# --------- Figure 2: Constructor DCSI time series ---------
team_df = race_df[race_df["constructor_name"] == team_name].copy()
team_df = team_df.sort_values(["season_id", "round"])

plt.figure(figsize=(7, 4))
for season, g in team_df.groupby("season_id"):
    plt.plot(
        g["round"],
        g["team_eff_mean"],
        marker="o",
        label=str(season)
    )

plt.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
plt.xlabel("Round")
plt.ylabel("Constructor Effect")
plt.title(f"Dynamic Constructor Strength Over Time – {team_name}")
plt.legend(title="Season", fontsize=8)
plt.tight_layout()

out_team_ts = OUT_DIR / "fig_constructor_dcsi_timeseries.png"
plt.savefig(out_team_ts, dpi=200)
plt.close()
print(f"[OK] Wrote {out_team_ts}")

# --------- Figure 3: Top “outperforming car” drivers ---------
if drv_rollup is not None and "outperforming_car" in drv_rollup.columns:
    top_drv = (
        drv_rollup.sort_values("outperforming_car", ascending=False)
        .head(10)
        .iloc[::-1]  # reverse for horizontal bar
    )

    plt.figure(figsize=(7, 4.5))
    plt.barh(top_drv["driver_name"], top_drv["outperforming_car"])
    plt.xlabel("Driver Effect – Team Effect")
    plt.ylabel("Driver")
    plt.title("Top 10 Drivers Outperforming Their Car (Pit+SC Model)")
    plt.tight_layout()

    out_outperf = OUT_DIR / "fig_outperforming_drivers.png"
    plt.savefig(out_outperf, dpi=200)
    plt.close()
    print(f"[OK] Wrote {out_outperf}")
else:
    print("[WARN] No rollup_drivers_overall with 'outperforming_car' found – skipping Figure 3.")
