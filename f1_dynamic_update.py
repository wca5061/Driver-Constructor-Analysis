# f1_dynamic_update.py
# Race-by-race Bayesian updating of Driver/Constructor effects + DCSI
# Uses sequential priors (posterior -> next prior) and writes race-level & cumulative CSVs.

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import arviz as az
import pymc as pm

# ---------------- Config ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# Inference settings
INFER_METHOD   = os.environ.get("F1_INFER", "advi")  # "advi" (fast) or "nuts"
ADVI_STEPS     = int(os.environ.get("F1_ADVI_STEPS", "4000"))
NUTS_DRAWS     = int(os.environ.get("F1_NUTS_DRAWS", "800"))
NUTS_TUNE      = int(os.environ.get("F1_NUTS_TUNE", "800"))
RNG_SEED       = int(os.environ.get("F1_SEED", "123"))

# Prior persistence (random-walk)
PRIOR_SD_DRIVER = float(os.environ.get("F1_PRIOR_SD_DRIVER", "0.20"))
PRIOR_SD_TEAM   = float(os.environ.get("F1_PRIOR_SD_TEAM", "0.15"))
PRIOR_SD_ALPHA  = 0.50
PRIOR_SD_BETA   = 0.25
LIKELIHOOD_SD_INIT = 0.20  # initial guess for perf noise

# ---------------- Load data ----------------
entries = pd.read_csv(os.path.join(DATA_DIR, "race_entries.csv"))
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
teams   = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
races   = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))

# Merge names & race metadata that we know exist
df = (entries
      .merge(drivers[["driver_id","driver_name"]], on="driver_id", how="left")
      .merge(teams[["constructor_id","constructor_name"]], on="constructor_id", how="left")
      .merge(races[["race_id","season_id","round","track_id","weather","laps_scheduled","sc_total","rain_mm"]],
             on="race_id", how="left"))

# Keep classified with valid gaps
df = df[(df["classified"] == True) & (~df["time_gap_s"].isna())].copy()

# ---- PERF: min-max per race (higher = better)
eps = 1e-6
g = df.groupby("race_id")["time_gap_s"]
df["perf"] = 1.0 - (df["time_gap_s"] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)
df["perf"] = df["perf"].clip(0.0, 1.0)

# Features (robust to missing)
df["street"] = df["street"].astype(int) if "street" in df.columns else 0
df["wet"]    = (df["weather"].fillna("").str.lower().str.contains("wet|mixed")).astype(int)
df["grid_c"] = df["grid"] - df.groupby("race_id")["grid"].transform("mean")

# ---------------- Chronological order (robust build) ----------------
# ensure date exists in races for sorting; synthesize if missing
if "date" not in races.columns:
    races = races.copy()
    races["date"] = pd.to_datetime(races["season_id"].astype(str) + "-01-01") + pd.to_timedelta(races["round"] * 14, unit="D")

race_order = (df[["race_id"]].drop_duplicates()
              .merge(races[["race_id","season_id","round","date"]], on="race_id", how="left"))
race_order = race_order.sort_values(["season_id","round"]).reset_index(drop=True)
race_ids = race_order["race_id"].tolist()
race_meta = race_order.set_index("race_id")[["season_id","round"]].to_dict("index")

# ---------------- Sequential priors & outputs ----------------
driver_prior = {}  # {driver_id: (mean, sd)}
team_prior   = {}  # {constructor_id: (mean, sd)}

rows_race = []     # per-race DCSI rows
driver_cum = {}    # {driver_id: [posterior means]}
team_cum   = {}    # {team_id: [posterior means]}

def get_prior(id_, prior_dict, default_sd):
    if id_ in prior_dict:
        m, s = prior_dict[id_]
    else:
        m, s = 0.0, default_sd
    s = max(s, 0.05)  # floor for stability
    return m, s

def update_prior(id_, post_mean, post_sd, prior_dict):
    prior_dict[id_] = (float(post_mean), float(max(post_sd, 0.05)))

def fit_one_race(rdf, rng_seed=RNG_SEED):
    """
    perf_i ~ Normal(alpha + theta_d[driver_i] + theta_t[team_i] + beta*X_i, sigma)
    Priors for theta_d/theta_t centered at previous posteriors for participants.
    """
    drv_ids = rdf["driver_id"].unique().tolist()
    tm_ids  = rdf["constructor_id"].unique().tolist()
    drv_index = {d:i for i,d in enumerate(drv_ids)}
    tm_index  = {t:i for i,t in enumerate(tm_ids)}

    X = rdf[["grid_c","street","wet"]].to_numpy(dtype=float)
    y = rdf["perf"].to_numpy(dtype=float)

    di = rdf["driver_id"].map(drv_index).to_numpy()
    ti = rdf["constructor_id"].map(tm_index).to_numpy()

    # sequential priors
    drv_mu0 = np.zeros(len(drv_ids)); drv_sd0 = np.zeros(len(drv_ids))
    for d,i in drv_index.items():
        m,s = get_prior(d, driver_prior, PRIOR_SD_DRIVER)
        drv_mu0[i], drv_sd0[i] = m, s

    tm_mu0 = np.zeros(len(tm_ids)); tm_sd0 = np.zeros(len(tm_ids))
    for t,i in tm_index.items():
        m,s = get_prior(t, team_prior, PRIOR_SD_TEAM)
        tm_mu0[i], tm_sd0[i] = m, s

    with pm.Model() as m:
        alpha = pm.Normal("alpha", mu=0.0, sigma=PRIOR_SD_ALPHA)
        beta  = pm.Normal("beta", mu=0.0, sigma=PRIOR_SD_BETA, shape=X.shape[1])

        theta_d = pm.Normal("theta_d", mu=drv_mu0, sigma=drv_sd0, shape=len(drv_ids))
        theta_t = pm.Normal("theta_t", mu=tm_mu0,  sigma=tm_sd0,  shape=len(tm_ids))

        sigma = pm.HalfNormal("sigma", sigma=LIKELIHOOD_SD_INIT)

        mu = alpha + theta_d[di] + theta_t[ti] + pm.math.dot(X, beta)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        if INFER_METHOD.lower() == "nuts":
            idata = pm.sample(draws=NUTS_DRAWS, tune=NUTS_TUNE, chains=2, target_accept=0.9,
                              random_seed=rng_seed, progressbar=False)
        else:
            approx = pm.fit(n=ADVI_STEPS, random_seed=rng_seed, progressbar=False)
            idata  = approx.sample(draws=1200, random_seed=rng_seed)

    post = idata.posterior
    drv_mean = post["theta_d"].mean(dim=["chain","draw"]).values
    drv_sd   = post["theta_d"].std(dim=["chain","draw"]).values
    tm_mean  = post["theta_t"].mean(dim=["chain","draw"]).values
    tm_sd    = post["theta_t"].std(dim=["chain","draw"]).values

    alpha_mean = float(post["alpha"].mean().values)
    beta_mean  = post["beta"].mean(dim=["chain","draw"]).values
    sigma_mean = float(post["sigma"].mean().values)

    mu_hat = alpha_mean + drv_mean[di] + tm_mean[ti] + (X @ beta_mean)

    # variance-based weights (simple proxy for decomposition)
    var_d = float(np.mean(drv_sd**2))
    var_t = float(np.mean(tm_sd**2))
    w_driver = var_d / (var_d + var_t + sigma_mean**2 + 1e-9)
    w_team   = 1.0 - w_driver

    return {
        "idata": idata,
        "drv_index": drv_index, "tm_index": tm_index,
        "drv_mean": drv_mean, "drv_sd": drv_sd,
        "tm_mean": tm_mean, "tm_sd": tm_sd,
        "alpha_mean": alpha_mean, "beta_mean": beta_mean, "sigma_mean": sigma_mean,
        "mu_hat": mu_hat, "w_driver": w_driver, "w_team": w_team
    }

# ---------------- Run sequentially ----------------
for rid in race_ids:
    rdf = df[df["race_id"] == rid].copy().reset_index(drop=True)
    if len(rdf) < 8:
        continue

    # Fit one race
    res = fit_one_race(rdf)

    # Update priors with posteriors (only those who raced)
    for d, idx in res["drv_index"].items():
        update_prior(d, res["drv_mean"][idx], res["drv_sd"][idx], driver_prior)
        driver_cum.setdefault(d, []).append(res["drv_mean"][idx])

    for t, idx in res["tm_index"].items():
        update_prior(t, res["tm_mean"][idx], res["tm_sd"][idx], team_prior)
        team_cum.setdefault(t, []).append(res["tm_mean"][idx])

    # Emit race-level rows (inject season/round from race_meta to avoid KeyErrors)
    meta = race_meta.get(rid, {"season_id": None, "round": None})
    tmp = rdf[[
        "race_id","driver_id","driver_name","constructor_id","constructor_name",
        "grid","finish_position","perf","street","wet"
    ]].copy()
    tmp["season_id"] = meta["season_id"]
    tmp["round"]     = meta["round"]
    tmp["pred_perf"] = res["mu_hat"]

    # attach posterior means used for decomposition
    di = rdf["driver_id"].map(res["drv_index"]).to_numpy()
    ti = rdf["constructor_id"].map(res["tm_index"]).to_numpy()
    tmp["driver_eff_mean"] = res["drv_mean"][di]
    tmp["team_eff_mean"]   = res["tm_mean"][ti]
    tmp["dcsi_weight_driver"] = res["w_driver"]
    tmp["dcsi_weight_team"]   = res["w_team"]

    # Per-entry normalized shares (|driver| vs |team|)
    total_abs = np.abs(tmp["driver_eff_mean"]) + np.abs(tmp["team_eff_mean"]) + 1e-9
    tmp["dcsi_driver_share_entry"] = np.abs(tmp["driver_eff_mean"]) / total_abs
    tmp["dcsi_team_share_entry"]   = np.abs(tmp["team_eff_mean"]) / total_abs

    rows_race.append(tmp)

# ---------------- Save outputs ----------------
race_out = pd.concat(rows_race, ignore_index=True) if rows_race else pd.DataFrame()
if len(race_out):
    race_out.sort_values(["season_id","round","driver_name"], inplace=True)
    race_out.to_csv(os.path.join(OUT_DIR, "dcsi_race.csv"), index=False)

    # Cumulative tables
    drv_means = {k: np.mean(v) for k,v in driver_cum.items()}
    team_means= {k: np.mean(v) for k,v in team_cum.items()}

    drv_df = (pd.DataFrame({"driver_id": list(drv_means.keys()),
                            "driver_eff_cum_mean": list(drv_means.values())})
              .merge(drivers[["driver_id","driver_name"]], on="driver_id", how="left")
              .sort_values("driver_eff_cum_mean", ascending=False))
    tm_df  = (pd.DataFrame({"constructor_id": list(team_means.keys()),
                            "team_eff_cum_mean": list(team_means.values())})
              .merge(teams[["constructor_id","constructor_name"]], on="constructor_id", how="left")
              .sort_values("team_eff_cum_mean", ascending=False))

    drv_df.to_csv(os.path.join(OUT_DIR, "dcsi_cumulative_drivers.csv"), index=False)
    tm_df.to_csv(os.path.join(OUT_DIR, "dcsi_cumulative_constructors.csv"), index=False)

    print("✅ Wrote:")
    print(" -", os.path.join(OUT_DIR, "dcsi_race.csv"))
    print(" -", os.path.join(OUT_DIR, "dcsi_cumulative_drivers.csv"))
    print(" -", os.path.join(OUT_DIR, "dcsi_cumulative_constructors.csv"))
else:
    print("No race outputs were produced. Check your input dataset and filters.")
