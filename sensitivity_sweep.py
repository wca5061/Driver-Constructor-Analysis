# sensitivity_sweep.py
# TRAIN-ONLY sensitivity & stress tests for the Dynamic DCSI pipeline.
# It NEVER reads TEST or VAL (enforced via splits.csv).
#
# Outputs -> {F1_OUT_DIR}/sensitivity/
#   - summary.csv (rank stability vs baseline)
#   - <variant>_drivers.csv
#   - <variant>_constructors.csv
#
# Env:
#   F1_DATA_DIR   (default: data/synth_f1_2018_2025_realish)
#   F1_OUT_DIR    (default: outputs/f1_dynamic_train)  # recommend pointing at your TRAIN folder
#   F1_SPLITS_CSV (default: outputs/splits/splits.csv)

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import kendalltau, spearmanr

# ---------------- Config ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic_train")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
SENS_DIR = os.path.join(OUT_DIR, "sensitivity")
Path(SENS_DIR).mkdir(parents=True, exist_ok=True)

SPLITS_CSV = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
if not os.path.exists(SPLITS_CSV):
    raise FileNotFoundError(f"Missing splits file: {SPLITS_CSV}")

# Baseline settings
BASE = dict(
    prior_sd_driver=0.20,
    prior_sd_team=0.15,
    infer="advi",         # "advi" | "nuts"
    advi_steps=2500,
    nuts_draws=600,
    nuts_tune=600,
    use_street=True,
    use_wet=True,
    use_gridc=True,
    season_min=None,
    season_max=None,
)

# Variants (all will be TRAIN-only)
VARIANTS = [
    dict(name="prior_d_0.10", prior_sd_driver=0.10),
    dict(name="prior_d_0.30", prior_sd_driver=0.30),
    dict(name="prior_t_0.10", prior_sd_team=0.10),
    dict(name="prior_t_0.25", prior_sd_team=0.25),

    dict(name="no_street", use_street=False),
    dict(name="no_wet",    use_wet=False),
    dict(name="no_gridc",  use_gridc=False),

    dict(name="nuts_recent", infer="nuts", season_min="2023", season_max="2023",
         nuts_draws=800, nuts_tune=800, advi_steps=None),
]

RNG_SEED = 123
LIKELIHOOD_SD_INIT = 0.20
PRIOR_SD_ALPHA = 0.50
PRIOR_SD_BETA  = 0.25

# ---------------- Load data ----------------
entries = pd.read_csv(os.path.join(DATA_DIR, "race_entries.csv"))
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
teams   = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
races   = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
splits  = pd.read_csv(SPLITS_CSV)[["race_id","split"]]

# Base df
df = (entries
      .merge(drivers[["driver_id","driver_name"]], on="driver_id", how="left")
      .merge(teams[["constructor_id","constructor_name"]], on="constructor_id", how="left")
      .merge(races[["race_id","season_id","round","track_id","weather","laps_scheduled","sc_total","rain_mm"]],
             on="race_id", how="left"))

df = df[(df["classified"] == True) & (~df["time_gap_s"].isna())].copy()

# PERF target (min-max per race)
eps = 1e-6
g = df.groupby("race_id")["time_gap_s"]
df["perf"] = 1.0 - (df["time_gap_s"] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)
df["perf"] = df["perf"].clip(0.0, 1.0)

# ensure date for sorting
if "date" not in races.columns:
    races = races.copy()
    races["date"] = pd.to_datetime(races["season_id"].astype(str) + "-01-01") + pd.to_timedelta(races["round"] * 14, unit="D")

df = df.merge(races[["race_id","date"]], on="race_id", how="left")

# Features
df["street"] = df["street"].astype(int) if "street" in df.columns else 0
df["wet"]    = (df["weather"].fillna("").str.lower().str.contains("wet|mixed")).astype(int)
df["grid_c"] = df["grid"] - df.groupby("race_id")["grid"].transform("mean")

# ---------------- Enforce TRAIN-only ----------------
df = df.merge(splits, on="race_id", how="left")
df = df[df["split"] == "TRAIN"].copy()
if df.empty:
    raise RuntimeError("TRAIN split has no rows. Check your splits.csv and data.")

# ---------------- Helpers ----------------
def rank_corr(df_a: pd.DataFrame, df_b: pd.DataFrame, key: str, score_col: str):
    a = df_a[[key, score_col]].dropna().copy()
    b = df_b[[key, score_col]].dropna().copy()
    m = a.merge(b, on=key, suffixes=("_a","_b"))
    if len(m) < 3:
        return np.nan, np.nan, int(len(m))
    m["rank_a"] = m[f"{score_col}_a"].rank(ascending=False, method="average")
    m["rank_b"] = m[f"{score_col}_b"].rank(ascending=False, method="average")
    tau, _ = kendalltau(m["rank_a"], m["rank_b"])
    rho, _ = spearmanr(m["rank_a"], m["rank_b"])
    return float(tau), float(rho), int(len(m))

def run_dynamic(config: dict):
    prior_sd_driver = config.get("prior_sd_driver", BASE["prior_sd_driver"])
    prior_sd_team   = config.get("prior_sd_team",   BASE["prior_sd_team"])
    infer           = config.get("infer",           BASE["infer"]).lower()
    advi_steps      = config.get("advi_steps",      BASE["advi_steps"])
    nuts_draws      = config.get("nuts_draws",      BASE["nuts_draws"])
    nuts_tune       = config.get("nuts_tune",       BASE["nuts_tune"])
    use_street      = config.get("use_street",      BASE["use_street"])
    use_wet         = config.get("use_wet",         BASE["use_wet"])
    use_gridc       = config.get("use_gridc",       BASE["use_gridc"])
    smin            = config.get("season_min",      BASE["season_min"])
    smax            = config.get("season_max",      BASE["season_max"])

    # make sure season_id exists
    d = df.copy()
    if "season_id" not in d.columns or "round" not in d.columns:
        races_fix = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))
        d = d.merge(races_fix[["race_id","season_id","round"]], on="race_id", how="left")

    # slice seasons
    if smin is not None and "season_id" in d.columns:
        d = d[d["season_id"].astype(str) >= str(smin)]
    if smax is not None and "season_id" in d.columns:
        d = d[d["season_id"].astype(str) <= str(smax)]

    # features
    feats = []
    if use_gridc: feats.append("grid_c")
    if use_street: feats.append("street")
    if use_wet: feats.append("wet")
    if not feats:
        feats = ["grid_c"]

    # race order
    race_order = (
        d[["race_id"]].drop_duplicates()
        .merge(races[["race_id","season_id","round","date"]], on="race_id", how="left")
        .sort_values(["season_id","round"])
        .reset_index(drop=True)
    )
    race_ids = race_order["race_id"].tolist()

    # sequential priors
    d_prior, t_prior = {}, {}
    d_cum, t_cum = {}, {}
    rng_seed = RNG_SEED

    for rid in race_ids:
        rdf = d[d["race_id"] == rid].copy().reset_index(drop=True)
        if len(rdf) < 8:
            continue

        drv_ids = rdf["driver_id"].unique().tolist()
        tm_ids  = rdf["constructor_id"].unique().tolist()
        di_map  = {x:i for i,x in enumerate(drv_ids)}
        ti_map  = {x:i for i,x in enumerate(tm_ids)}

        X = rdf[feats].to_numpy(dtype=float)
        y = rdf["perf"].to_numpy(dtype=float)
        di = rdf["driver_id"].map(di_map).to_numpy()
        ti = rdf["constructor_id"].map(ti_map).to_numpy()

        drv_mu0 = np.array([d_prior.get(did, (0.0, prior_sd_driver))[0] for did in drv_ids])
        drv_sd0 = np.array([max(d_prior.get(did, (0.0, prior_sd_driver))[1], 0.05) for did in drv_ids])
        tm_mu0  = np.array([t_prior.get(tid, (0.0, prior_sd_team))[0] for tid in tm_ids])
        tm_sd0  = np.array([max(t_prior.get(tid, (0.0, prior_sd_team))[1], 0.05) for tid in tm_ids])

        with pm.Model() as m:
            alpha = pm.Normal("alpha", mu=0.0, sigma=PRIOR_SD_ALPHA)
            beta  = pm.Normal("beta",  mu=0.0, sigma=PRIOR_SD_BETA, shape=X.shape[1])
            theta_d = pm.Normal("theta_d", mu=drv_mu0, sigma=drv_sd0, shape=len(drv_ids))
            theta_t = pm.Normal("theta_t", mu=tm_mu0,  sigma=tm_sd0,  shape=len(tm_ids))
            sigma = pm.HalfNormal("sigma", sigma=LIKELIHOOD_SD_INIT)

            mu = alpha + theta_d[di] + theta_t[ti] + pm.math.dot(X, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

            if infer == "nuts":
                idata = pm.sample(draws=nuts_draws, tune=nuts_tune, chains=2,
                                  target_accept=0.9, random_seed=rng_seed,
                                  progressbar=False)
            else:
                approx = pm.fit(n=advi_steps, random_seed=rng_seed, progressbar=False)
                idata  = approx.sample(draws=1000, random_seed=rng_seed)

        post = idata.posterior
        drv_mean = post["theta_d"].mean(dim=["chain","draw"]).values
        drv_sd   = post["theta_d"].std(dim=["chain","draw"]).values
        tm_mean  = post["theta_t"].mean(dim=["chain","draw"]).values
        tm_sd    = post["theta_t"].std(dim=["chain","draw"]).values

        for d_id, idx in di_map.items():
            d_prior[d_id] = (float(drv_mean[idx]), float(max(drv_sd[idx], 0.05)))
            d_cum.setdefault(d_id, []).append(drv_mean[idx])
        for t_id, idx in ti_map.items():
            t_prior[t_id] = (float(tm_mean[idx]), float(max(tm_sd[idx], 0.05)))
            t_cum.setdefault(t_id, []).append(tm_mean[idx])

    drv_df = pd.DataFrame({
        "driver_id": list(d_cum.keys()),
        "driver_eff_cum_mean": [float(np.mean(v)) for v in d_cum.values()]
    }).merge(drivers[["driver_id","driver_name"]], on="driver_id", how="left") \
     .sort_values("driver_eff_cum_mean", ascending=False)

    tm_df = pd.DataFrame({
        "constructor_id": list(t_cum.keys()),
        "team_eff_cum_mean": [float(np.mean(v)) for v in t_cum.values()]
    }).merge(teams[["constructor_id","constructor_name"]], on="constructor_id", how="left") \
     .sort_values("team_eff_cum_mean", ascending=False)

    return drv_df.reset_index(drop=True), tm_df.reset_index(drop=True)

def save_variant(name, drv_df, tm_df):
    drv_path = os.path.join(SENS_DIR, f"{name}_drivers.csv")
    tm_path  = os.path.join(SENS_DIR, f"{name}_constructors.csv")
    drv_df.to_csv(drv_path, index=False)
    tm_df.to_csv(tm_path, index=False)
    return drv_path, tm_path

# ---------------- Run baseline ----------------
print("[INFO] Running TRAIN-only sensitivity sweep")
drv_base, tm_base = run_dynamic(BASE)
save_variant("baseline", drv_base, tm_base)

# ---------------- Run variants & compare ----------------
rows_summary = []
for v in VARIANTS:
    cfg = BASE.copy(); cfg.update(v)
    vname = v.get("name") or f"variant_{len(rows_summary)+1}"
    print(f"[INFO] Variant: {vname}")
    drv_v, tm_v = run_dynamic(cfg)
    save_variant(vname, drv_v, tm_v)

    tau_d, rho_d, n_d = rank_corr(drv_base, drv_v, "driver_id", "driver_eff_cum_mean")
    tau_t, rho_t, n_t = rank_corr(tm_base, tm_v, "constructor_id", "team_eff_cum_mean")

    rows_summary.append({
        "variant": vname,
        "prior_sd_driver": cfg["prior_sd_driver"],
        "prior_sd_team":   cfg["prior_sd_team"],
        "infer":           cfg["infer"],
        "use_street":      cfg["use_street"],
        "use_wet":         cfg["use_wet"],
        "use_gridc":       cfg["use_gridc"],
        "season_min":      cfg["season_min"],
        "season_max":      cfg["season_max"],
        "drivers_kendall_tau": tau_d,
        "drivers_spearman_rho": rho_d,
        "drivers_overlap_n": n_d,
        "constructors_kendall_tau": tau_t,
        "constructors_spearman_rho": rho_t,
        "constructors_overlap_n": n_t,
    })

summary = pd.DataFrame(rows_summary)
summary_path = os.path.join(SENS_DIR, "summary.csv")
summary.to_csv(summary_path, index=False)

print("\n✅ Sensitivity sweep complete (TRAIN only).")
print(" - Summary:", summary_path)
print(" - Baseline:", os.path.join(SENS_DIR, "baseline_drivers.csv"),
      " & ", os.path.join(SENS_DIR, "baseline_constructors.csv"))
print(" - Variants saved alongside baseline in:", SENS_DIR)
