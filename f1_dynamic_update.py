# f1_dynamic_update.py
# Dynamic Bayesian DCSI with split-aware TRAIN/TEST/VAL behavior.
# Defaults:
#   TRAIN -> FIT (sequential MCMC/ADVI) + save priors
#   TEST  -> PREDICT-ONLY using TRAIN priors (no fitting, no updates)
#   VAL   -> PREDICT-ONLY using TRAIN priors (no fitting, no updates)
#
# Env flags:
#   F1_DATA_DIR       (default: data/synth_f1_2018_2025_realish)
#   F1_OUT_DIR        (default: outputs/f1_dynamic)
#   F1_SPLITS_CSV     (default: outputs/splits/splits.csv)
#   F1_SPLIT_TARGET   TRAIN | TEST | VAL   (default: TRAIN)
#   F1_RUN_MODE       fit | predict        (default: auto: TRAIN->fit, TEST/VAL->predict)
#   F1_SAVE_PRIORS    1/0                  (default: 1 for TRAIN, ignored otherwise)
#   F1_PRIORS_PATH    (default: {OUT_DIR}/priors.npz)  # where to save/load priors
#
# Inference (fit mode only):
#   F1_INFER          advi|nuts (default advi)
#   F1_ADVI_STEPS     (default 4000)
#   F1_NUTS_DRAWS     (default 800)
#   F1_NUTS_TUNE      (default 800)
#   F1_SEED           (default 123)
#
# Priors:
#   F1_PRIOR_SD_DRIVER (default 0.20)
#   F1_PRIOR_SD_TEAM   (default 0.15)

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pymc as pm

# ---------------- Config ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

SPLITS_CSV    = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
SPLIT_TARGET  = os.environ.get("F1_SPLIT_TARGET", "TRAIN").upper()   # TRAIN | TEST | VAL
RUN_MODE_ENV  = os.environ.get("F1_RUN_MODE", "").lower()            # "fit" | "predict" | ""
SAVE_PRIORS   = os.environ.get("F1_SAVE_PRIORS", "1") == "1"
PRIORS_PATH   = os.environ.get("F1_PRIORS_PATH", os.path.join(OUT_DIR, "priors.npz"))

# Inference settings (for FIT mode)
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

# ---------------- Load core data ----------------
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

# Keep classified with valid gaps (we still compute perf in predict mode for evaluation)
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

# ---------------- Chronological order (robust) ----------------
# ensure date exists in races for sorting; synthesize if missing
if "date" not in races.columns:
    races = races.copy()
    races["date"] = pd.to_datetime(races["season_id"].astype(str) + "-01-01") + pd.to_timedelta(races["round"] * 14, unit="D")

race_order = (df[["race_id"]].drop_duplicates()
              .merge(races[["race_id","season_id","round","date"]], on="race_id", how="left"))

# ---------------- Apply split (and never read VAL unless explicitly targeted) ----------------
if not os.path.exists(SPLITS_CSV):
    raise FileNotFoundError(f"Missing split file: {SPLITS_CSV}")

splits = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
race_order = race_order.merge(splits, on="race_id", how="left")

# Reject VAL unless requested
if SPLIT_TARGET not in {"TRAIN","TEST","VAL"}:
    raise ValueError("F1_SPLIT_TARGET must be TRAIN, TEST, or VAL")

race_order = race_order[race_order["split"].isin(["TRAIN","TEST","VAL"])]
race_order = race_order[race_order["split"] == SPLIT_TARGET]
race_order = race_order.sort_values(["season_id","round"]).reset_index(drop=True)

race_ids = race_order["race_id"].tolist()
race_meta = race_order.set_index("race_id")[["season_id","round"]].to_dict("index")
print(f"[INFO] Split={SPLIT_TARGET} -> {len(race_ids)} races")

# ---------------- Decide run mode (fit vs predict) ----------------
if RUN_MODE_ENV in {"fit","predict"}:
    RUN_MODE = RUN_MODE_ENV
else:
    # default policy: TRAIN=fit, TEST/VAL=predict
    RUN_MODE = "fit" if SPLIT_TARGET == "TRAIN" else "predict"

print(f"[INFO] Run mode resolved to: {RUN_MODE}")

# ---------------- Utilities ----------------
def save_preds_csv(rows, fname):
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out.sort_values(["season_id","round","driver_name"], inplace=True)
        path = os.path.join(OUT_DIR, fname)
        out.to_csv(path, index=False)
        print(f"✅ Wrote: {path}")
    else:
        print("No outputs produced.")

# ---------------- PREDICT-ONLY path (TEST/VAL by default) ----------------
if RUN_MODE == "predict":
    # Load priors learned on TRAIN
    if not os.path.exists(PRIORS_PATH):
        raise FileNotFoundError(f"Missing priors from TRAIN: {PRIORS_PATH}")

    npz = np.load(PRIORS_PATH, allow_pickle=True)
    drv_mu_map = {d: m for d, m in zip(npz["drv_ids"], npz["drv_mus"])}
    tm_mu_map  = {t: m for t, m in zip(npz["tm_ids"],  npz["tm_mus"])}
    alpha0     = float(npz.get("alpha_mean_global", 0.0))
    beta0      = npz.get("beta_mean_global", np.zeros(3))

    # align beta to [grid_c, street, wet]
    if beta0.shape[0] != 3:
        beta0 = np.zeros(3)

    rows = []
    for rid in race_ids:
        rdf = df[df["race_id"] == rid].copy().reset_index(drop=True)
        if len(rdf) < 8:
            continue

        X = rdf[["grid_c","street","wet"]].to_numpy(float)
        drv = rdf["driver_id"].map(drv_mu_map).fillna(0.0).to_numpy(float)
        tm  = rdf["constructor_id"].map(tm_mu_map).fillna(0.0).to_numpy(float)
        pred_perf = alpha0 + drv + tm + (X @ beta0)

        meta = race_meta.get(rid, {"season_id": None, "round": None})
        tmp = rdf[[
            "race_id","driver_id","driver_name","constructor_id","constructor_name",
            "grid","finish_position","perf","street","wet"
        ]].copy()
        tmp["season_id"] = meta["season_id"]
        tmp["round"]     = meta["round"]
        tmp["pred_perf"] = pred_perf

        # Optional: expose the components used for explainability
        tmp["driver_eff_mean"] = drv
        tmp["team_eff_mean"]   = tm
        rows.append(tmp)

    # Write predictions to the same filename the downstream scripts expect
    save_preds_csv(rows, "dcsi_race.csv")
    # No cumulative tables in predict-only mode (to avoid "learning" on holdout)
    raise SystemExit(0)

# ---------------- FIT path (TRAIN by default) ----------------
# Sequential priors & outputs
driver_prior = {}  # {driver_id: (mean, sd)}
team_prior   = {}  # {constructor_id: (mean, sd)}

rows_race = []     # per-race DCSI rows
driver_cum = {}    # {driver_id: [posterior means]}
team_cum   = {}    # {team_id: [posterior means]}

alpha_hist = []
beta_hist  = []

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

    # record for global averages
    alpha_hist.append(alpha_mean)
    beta_hist.append(beta_mean)

    return {
        "drv_index": {**drv_index}, "tm_index": {**tm_index},
        "drv_mean": drv_mean, "drv_sd": drv_sd,
        "tm_mean": tm_mean, "tm_sd": tm_sd,
        "alpha_mean": alpha_mean, "beta_mean": beta_mean, "sigma_mean": sigma_mean,
        "mu_hat": mu_hat
    }

# ---------------- Train sequentially ----------------
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

    # Emit race-level rows (inject season/round)
    meta = race_meta.get(rid, {"season_id": None, "round": None})
    tmp = rdf[[
        "race_id","driver_id","driver_name","constructor_id","constructor_name",
        "grid","finish_position","perf","street","wet"
    ]].copy()
    tmp["season_id"] = meta["season_id"]
    tmp["round"]     = meta["round"]
    tmp["pred_perf"] = res["mu_hat"]

    # attach posterior means (for explainability and later comparisons)
    di = rdf["driver_id"].map(res["drv_index"]).to_numpy()
    ti = rdf["constructor_id"].map(res["tm_index"]).to_numpy()
    tmp["driver_eff_mean"] = res["drv_mean"][di]
    tmp["team_eff_mean"]   = res["tm_mean"][ti]

    # Per-entry normalized shares (|driver| vs |team|)
    total_abs = np.abs(tmp["driver_eff_mean"]) + np.abs(tmp["team_eff_mean"]) + 1e-9
    tmp["dcsi_driver_share_entry"] = np.abs(tmp["driver_eff_mean"]) / total_abs
    tmp["dcsi_team_share_entry"]   = np.abs(tmp["team_eff_mean"]) / total_abs

    rows_race.append(tmp)

# ---------------- Save TRAIN outputs ----------------
save_preds_csv(rows_race, "dcsi_race.csv")

# Cumulative tables (TRAIN only)
race_out = pd.concat(rows_race, ignore_index=True) if rows_race else pd.DataFrame()
if len(race_out):
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
    print(" -", os.path.join(OUT_DIR, "dcsi_cumulative_drivers.csv"))
    print(" -", os.path.join(OUT_DIR, "dcsi_cumulative_constructors.csv"))

# ---------------- Save priors for holdout prediction ----------------
if SAVE_PRIORS and SPLIT_TARGET == "TRAIN":
    alpha_mean_global = float(np.mean(alpha_hist)) if alpha_hist else 0.0
    beta_mean_global  = np.mean(beta_hist, axis=0) if len(beta_hist) else np.zeros(3)

    drv_ids = np.array(list(driver_prior.keys()))
    drv_mus = np.array([driver_prior[k][0] for k in drv_ids])
    drv_sds = np.array([driver_prior[k][1] for k in drv_ids])

    tm_ids  = np.array(list(team_prior.keys()))
    tm_mus  = np.array([team_prior[k][0] for k in tm_ids])
    tm_sds  = np.array([team_prior[k][1] for k in tm_ids])

    np.savez(PRIORS_PATH,
             drv_ids=drv_ids, drv_mus=drv_mus, drv_sds=drv_sds,
             tm_ids=tm_ids, tm_mus=tm_mus, tm_sds=tm_sds,
             alpha_mean_global=alpha_mean_global,
             beta_mean_global=beta_mean_global)
    print(f"💾 Saved priors to {PRIORS_PATH}")
