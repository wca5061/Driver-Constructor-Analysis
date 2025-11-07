# export_dashboard_json.py
#
# Collects dashboard-ready JSON from dynamic F1 outputs.
#
# Env (optional):
#   F1_DASH_IN_DIR      = base output dir (default: outputs/f1_dynamic_train_pit_sc)
#   F1_DASH_DUALITY_DIR = dir where duality_metrics.json lives (default: F1_DASH_IN_DIR)
#   F1_DASH_OUT_PATH    = output JSON path (default: outputs/dashboard/dashboard_export.json)

import os
import json
from pathlib import Path

import pandas as pd

# --------- Config ---------

IN_DIR = Path(os.environ.get("F1_DASH_IN_DIR", "outputs/f1_dynamic_train_pit_sc"))
DUALITY_DIR = Path(os.environ.get("F1_DASH_DUALITY_DIR", str(IN_DIR)))
OUT_PATH = Path(os.environ.get("F1_DASH_OUT_PATH", "outputs/dashboard/dashboard_export.json"))

print(f"[INFO] Using IN_DIR      = {IN_DIR}")
print(f"[INFO] Using DUALITY_DIR = {DUALITY_DIR}")
print(f"[INFO] Will write JSON   = {OUT_PATH}")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------- Helpers ---------

def safe_read_csv(path: Path):
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return None
    return pd.read_csv(path)

def safe_read_json(path: Path):
    if not path.exists():
        print(f"[WARN] JSON not found: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def round_float(x, ndigits=4):
    try:
        return round(float(x), ndigits)
    except Exception:
        return x

# --------- 1. Top “Outperforming Car” drivers ---------

drivers_rollup_path = IN_DIR / "rollup_drivers_overall.csv"
drv_df = safe_read_csv(drivers_rollup_path)

top_outperforming_drivers = []
if drv_df is not None:
    drv_df = drv_df.sort_values("outperforming_car", ascending=False).head(10)
    for _, row in drv_df.iterrows():
        top_outperforming_drivers.append({
            "driver_id": row["driver_id"],
            "driver_name": row["driver_name"],
            "outperforming_car": round_float(row["outperforming_car"]),
            "driver_share_mean": round_float(row.get("driver_share_mean", None)),
            "team_share_mean": round_float(row.get("team_share_mean", None)),
            "starts": int(row.get("starts", 0)),
            "p_win_mean": round_float(row.get("p_win_mean", None)),
        })

# --------- 2. Team strength rankings ---------

teams_rollup_path = IN_DIR / "rollup_teams_overall.csv"
tm_df = safe_read_csv(teams_rollup_path)

team_strength_rankings = []
if tm_df is not None:
    tm_df = tm_df.sort_values("team_strength", ascending=False)
    for _, row in tm_df.iterrows():
        team_strength_rankings.append({
            "constructor_id": row["constructor_id"],
            "constructor_name": row["constructor_name"],
            "team_strength": round_float(row["team_strength"]),
            "avg_pred_perf": round_float(row.get("avg_pred_perf", None)),
            "starts": int(row.get("starts", 0)),
        })

# --------- 3. DCSI time series (driver & constructor) ---------

dcsi_race_path = IN_DIR / "dcsi_race.csv"
race_df = safe_read_csv(dcsi_race_path)

driver_dcsi_ts = []
team_dcsi_ts = []

if race_df is not None:
    # Driver-level time series (one row per race+driver)
    for _, row in race_df.iterrows():
        driver_dcsi_ts.append({
            "race_id": row["race_id"],
            "season_id": int(row.get("season_id", 0)),
            "round": int(row.get("round", 0)),
            "driver_id": row["driver_id"],
            "driver_name": row["driver_name"],
            "constructor_id": row["constructor_id"],
            "constructor_name": row["constructor_name"],
            "finish_position": int(row.get("finish_position", 0)),
            "pred_perf": round_float(row.get("pred_perf", None)),
            "driver_eff_mean": round_float(row.get("driver_eff_mean", None)),
            "team_eff_mean": round_float(row.get("team_eff_mean", None)),
            "dcsi_driver_share_entry": round_float(row.get("dcsi_driver_share_entry", None)),
            "dcsi_team_share_entry": round_float(row.get("dcsi_team_share_entry", None)),
        })

    # Constructor-level time series: aggregate team effect per race
    # Replace the existing team_dcsi_ts block with this:
    grp = (
        race_df
        .groupby(
            ["season_id", "constructor_id", "constructor_name"],
            observed=True
        )["team_eff_mean"]
        .mean()
        .reset_index()
    )
    for _, row in grp.iterrows():
        team_dcsi_ts.append({
            "season_id": int(row["season_id"]),
            "constructor_id": row["constructor_id"],
            "constructor_name": row["constructor_name"],
            "team_eff_mean": round_float(row["team_eff_mean"]),
        })
# --------- 4. Calibration metrics ---------

calibration_metrics_path = IN_DIR / "prob_metrics.json"
calibration_metrics = safe_read_json(calibration_metrics_path)

# --------- 5. Duality metrics ---------

duality_metrics_path = DUALITY_DIR / "duality_metrics.json"
duality_metrics = safe_read_json(duality_metrics_path)

# --------- 6. Build final payload ---------

dashboard = {
    "meta": {
        "in_dir": str(IN_DIR),
        "duality_dir": str(DUALITY_DIR),
    },
    "top_outperforming_drivers": top_outperforming_drivers,
    "team_strength_rankings": team_strength_rankings,
    "driver_dcsi_timeseries": driver_dcsi_ts,
    "team_dcsi_timeseries": team_dcsi_ts,
    "calibration_metrics": calibration_metrics,
    "duality_metrics": duality_metrics,
}

with open(OUT_PATH, "w") as f:
    json.dump(dashboard, f, indent=2)

print(f"✅ Wrote dashboard JSON to: {OUT_PATH}")
print(f"   Sections:")
for k in dashboard.keys():
    print(f"   - {k}")
