# f1_diagnostics.py (split-aware)
# Produces diagnostics for either:
#  1) Parent model (posterior.nc + posterior_*.csv), if present, OR
#  2) Dynamic pipeline (dcsi_race.csv + optional dcsi_probs.csv)
#
# Split-aware: honors F1_SPLIT_TARGET (TRAIN/TEST/VAL) and never includes VAL unless explicitly set.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic")  # point to split folder for dynamic mode
FIG_DIR  = os.path.join(OUT_DIR, "figs")
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

# Common helpers
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def apply_split(df, key="race_id"):
    if not os.path.exists(SPLITS_CSV) or key not in df.columns:
        return df
    s = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
    out = df.merge(s, on="race_id", how="left")
    if SPLIT_TARGET in {"TRAIN","TEST","VAL"}:
        out = out[out["split"] == SPLIT_TARGET]
    else:
        out = out[out["split"].isin(["TRAIN","TEST"])]  # default safeguard
    return out

# ---------------- Detect mode ----------------
posterior_path = os.path.join(OUT_DIR, "posterior.nc")
dynamic_csv    = os.path.join(OUT_DIR, "dcsi_race.csv")

PARENT_MODE  = os.path.exists(posterior_path)
DYNAMIC_MODE = os.path.exists(dynamic_csv)

if not (PARENT_MODE or DYNAMIC_MODE):
    raise FileNotFoundError(
        f"Neither parent posterior nor dynamic outputs found in {OUT_DIR}.\n"
        f"- Expected one of:\n  {posterior_path}\n  {dynamic_csv}"
    )

# ============================================================
# =============== 1) Parent-model diagnostics ================
# ============================================================
if PARENT_MODE:
    print("[INFO] Parent-model diagnostics mode")
    idata = az.from_netcdf(posterior_path)

    drv = pd.read_csv(os.path.join(OUT_DIR, "posterior_drivers.csv"))
    tm  = pd.read_csv(os.path.join(OUT_DIR, "posterior_constructors.csv"))
    rc  = pd.read_csv(os.path.join(OUT_DIR, "posterior_races.csv"))
    fx  = pd.read_csv(os.path.join(OUT_DIR, "posterior_coefs.csv"))

    # Load base race data (to compute perf + join splits)
    entries = pd.read_csv(os.path.join(DATA_DIR, "race_entries.csv"))
    entries = entries[(entries["classified"] == True) & (~entries["time_gap_s"].isna())].copy()

    # PERF min-max per race
    eps = 1e-6
    g = entries.groupby("race_id", observed=True)["time_gap_s"]
    entries["perf"] = 1.0 - (entries["time_gap_s"] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)
    entries["perf"] = entries["perf"].clip(0.0, 1.0)

    # apply split if available
    entries = apply_split(entries)

    # Trace plots
    az.plot_trace(
        idata,
        var_names=["alpha", "beta", "sigma_driver", "sigma_team", "sigma_race_i"],
        compact=True
    )
    savefig(os.path.join(FIG_DIR, f"trace_core_{SPLIT_TARGET}.png"))

    # Forest: top drivers
    TOP_N = 20
    top_drv = drv.sort_values("mean", ascending=False).head(TOP_N).copy()[::-1]
    plt.figure(figsize=(7, 10))
    ypos = np.arange(len(top_drv))
    plt.errorbar(
        x=top_drv["mean"], y=ypos,
        xerr=[top_drv["mean"] - top_drv["hdi_3%"], top_drv["hdi_97%"] - top_drv["mean"]],
        fmt="o", capsize=3
    )
    plt.yticks(ypos, top_drv["driver_id"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Driver Effect (posterior mean, 94% HDI)")
    plt.title(f"{SPLIT_TARGET}: Top {TOP_N} Drivers")
    savefig(os.path.join(FIG_DIR, f"drivers_forest_top_{SPLIT_TARGET}.png"))

    # Forest: top constructors
    top_tm = tm.sort_values("mean", ascending=False).head(TOP_N).copy()[::-1]
    plt.figure(figsize=(7, 10))
    ypos = np.arange(len(top_tm))
    plt.errorbar(
        x=top_tm["mean"], y=ypos,
        xerr=[top_tm["mean"] - top_tm["hdi_3%"], top_tm["hdi_97%"] - top_tm["mean"]],
        fmt="o", capsize=3
    )
    plt.yticks(ypos, top_tm["constructor_id"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("Constructor Effect (posterior mean, 94% HDI)")
    plt.title(f"{SPLIT_TARGET}: Top {TOP_N} Constructors")
    savefig(os.path.join(FIG_DIR, f"constructors_forest_top_{SPLIT_TARGET}.png"))

    # Merge readable names for scatter
    drivers_map = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))[["driver_id","driver_name"]]
    constructors_map = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))[["constructor_id","constructor_name"]]

    drv_mean = drv[["driver_id","mean"]].rename(columns={"mean":"driver_eff"}).merge(drivers_map, on="driver_id", how="left")
    tm_mean  = tm[["constructor_id","mean"]].rename(columns={"mean":"team_eff"}).merge(constructors_map, on="constructor_id", how="left")

    merged = (entries
              .merge(drv_mean[["driver_id","driver_eff","driver_name"]], on="driver_id", how="left")
              .merge(tm_mean[["constructor_id","team_eff","constructor_name"]], on="constructor_id", how="left"))

    by_driver = (merged.groupby(["driver_id","driver_name"], as_index=False)[["driver_eff","team_eff","perf"]].mean())

    # Scatter: driver vs constructor
    plt.figure(figsize=(7,7))
    plt.scatter(by_driver["team_eff"], by_driver["driver_eff"], s=30, alpha=0.8)
    for _, r in by_driver.sort_values("driver_eff", ascending=False).head(12).iterrows():
        plt.text(r["team_eff"], r["driver_eff"], r["driver_name"].split()[-1], fontsize=8, va="bottom")
    plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
    plt.xlabel("Constructor Effect (mean)"); plt.ylabel("Driver Effect (mean)")
    plt.title(f"{SPLIT_TARGET}: Driver vs Constructor Effects (Parent Model)")
    savefig(os.path.join(FIG_DIR, f"driver_vs_constructor_parent_{SPLIT_TARGET}.png"))

    # Outperforming-car bar
    by_driver["outperforming_car"] = by_driver["driver_eff"] - by_driver["team_eff"]
    top = by_driver.sort_values("outperforming_car", ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(8,7))
    plt.barh(top["driver_name"], top["outperforming_car"])
    plt.xlabel("Driver Effect – Team Effect (higher = punching above car)")
    plt.title(f"{SPLIT_TARGET}: Top 20 Outperformers (Parent Model)")
    savefig(os.path.join(FIG_DIR, f"drivers_outperforming_parent_{SPLIT_TARGET}.png"))

    print(f"✅ Parent-model diagnostics saved to: {FIG_DIR}")

# ============================================================
# =============== 2) Dynamic-pipeline diagnostics ============
# ============================================================
if DYNAMIC_MODE:
    print("[INFO] Dynamic-pipeline diagnostics mode")
    r = pd.read_csv(dynamic_csv)
    r = apply_split(r)

    # required columns
    req = ["race_id","driver_id","driver_name","constructor_id","constructor_name",
           "season_id","round","pred_perf"]
    for c in req:
        if c not in r.columns:
            raise KeyError(f"Required column missing in dcsi_race.csv: {c}")

    # perf proxy if present
    if "perf" not in r.columns and {"time_gap_s","race_id"}.issubset(set(r.columns)):
        eps = 1e-6
        g = r.groupby("race_id")["time_gap_s"]
        r["perf"] = 1.0 - (r["time_gap_s"] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)
        r["perf"] = r["perf"].clip(0.0, 1.0)

    # robustness for flags
    r["street"] = r["street"] if "street" in r.columns else 0
    r["wet"]    = r["wet"]    if "wet"    in r.columns else 0

    # Driver vs Constructor scatter (posterior means from dynamic outputs)
    if {"driver_eff_mean","team_eff_mean"}.issubset(set(r.columns)):
        drv_scatter = (r.groupby(["driver_id","driver_name"], as_index=False)
                         .agg(driver_eff=("driver_eff_mean","mean"),
                              team_eff=("team_eff_mean","mean")))
        plt.figure(figsize=(7,7))
        plt.scatter(drv_scatter["team_eff"], drv_scatter["driver_eff"], s=30, alpha=0.8)
        for _, row in drv_scatter.sort_values("driver_eff", ascending=False).head(12).iterrows():
            plt.text(row["team_eff"], row["driver_eff"], row["driver_name"].split()[-1], fontsize=8, va="bottom")
        plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
        plt.xlabel("Constructor Effect (mean)"); plt.ylabel("Driver Effect (mean)")
        plt.title(f"{SPLIT_TARGET}: Driver vs Constructor Effects (Dynamic)")
        savefig(os.path.join(FIG_DIR, f"driver_vs_constructor_dynamic_{SPLIT_TARGET}.png"))

        # Outperforming car ranking
        drv_scatter["delta"] = drv_scatter["driver_eff"] - drv_scatter["team_eff"]
        top = drv_scatter.sort_values("delta", ascending=False).head(20).iloc[::-1]
        plt.figure(figsize=(8,7))
        plt.barh(top["driver_name"], top["delta"])
        plt.xlabel("Driver Effect – Team Effect (mean)")
        plt.title(f"{SPLIT_TARGET}: Top 20 Outperformers (Dynamic)")
        savefig(os.path.join(FIG_DIR, f"drivers_outperforming_dynamic_{SPLIT_TARGET}.png"))

    # Calibration scatter: predicted rank vs actual finish (if finish_position present)
    if "finish_position" in r.columns:
        cal = r[["race_id","driver_id","pred_perf","finish_position"]].copy()
        cal["pred_rank"] = cal.groupby("race_id")["pred_perf"].rank(ascending=False, method="average")
        plt.figure(figsize=(6,6))
        plt.scatter(cal["pred_rank"], cal["finish_position"], s=10, alpha=0.5)
        lims = [0, cal[["pred_rank","finish_position"]].max().max() + 1]
        plt.plot(lims, lims, linestyle="--")
        plt.xlim(0, lims[1]); plt.ylim(0, lims[1])
        plt.xlabel("Predicted Rank"); plt.ylabel("Actual Finish Position")
        plt.title(f"{SPLIT_TARGET}: Calibration — Predicted Rank vs Finish")
        savefig(os.path.join(FIG_DIR, f"calibration_predrank_vs_finish_{SPLIT_TARGET}.png"))

    # If probabilities exist, plot reliability (mini)
    probs_path = os.path.join(OUT_DIR, "dcsi_probs.csv")
    if os.path.exists(probs_path):
        p = pd.read_csv(probs_path)
        p = apply_split(p)
        if set(["race_id","driver_id","p_win"]).issubset(p.columns) and "finish_position" in r.columns:
            p = p.merge(r[["race_id","driver_id","finish_position"]], on=["race_id","driver_id"], how="left")
            p["win"] = (p.groupby("race_id")["finish_position"].transform("min") == p["finish_position"]).astype(int)

            # decile reliability
            q = pd.qcut(p["p_win"], q=10, labels=False, duplicates="drop")
            grp = p.groupby(q, observed=True).agg(p_mean=("p_win","mean"), y_rate=("win","mean")).reset_index(drop=True)
            plt.figure(figsize=(5.2,4.2))
            plt.plot(grp["p_mean"], grp["y_rate"], marker="o")
            plt.plot([0,1],[0,1],"--",linewidth=1)
            plt.xlabel("Predicted win prob (bin mean)"); plt.ylabel("Empirical win rate")
            plt.title(f"{SPLIT_TARGET}: Reliability — Win Prob")
            savefig(os.path.join(FIG_DIR, f"reliability_win_{SPLIT_TARGET}.png"))

    print(f"✅ Dynamic diagnostics saved to: {FIG_DIR}")