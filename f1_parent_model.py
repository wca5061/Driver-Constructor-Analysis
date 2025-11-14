# f1_parent_model.py (split-aware)
# Parent hierarchical model for F1:
#   perf ~ alpha + driver_effect + constructor_effect + X*beta + eps
# Split policy:
#   TRAIN -> FIT only on TRAIN, save posterior (+ posterior_*.csv)
#   TEST/VAL -> PREDICT-ONLY using posterior means saved from TRAIN (no fitting)
#
# Env:
#   F1_DATA_DIR       default: data/synth_f1_2018_2025_realish
#   F1_OUT_DIR        default: outputs/f1_parent_model
#   F1_SPLITS_CSV     default: outputs/splits/splits.csv
#   F1_SPLIT_TARGET   TRAIN | TEST | VAL   (default TRAIN)
#   F1_RUN_MODE       fit | predict        (auto: TRAIN->fit, TEST/VAL->predict)
#   F1_INFER          advi|nuts (default advi)
#   F1_ADVI_STEPS     default 6000
#   F1_NUTS_DRAWS     default 1000
#   F1_NUTS_TUNE      default 1000
#   F1_SEED           default 123
#
# Outputs (TRAIN fit):
#   {OUT}/posterior.nc
#   {OUT}/posterior_drivers.csv
#   {OUT}/posterior_constructors.csv
#   {OUT}/posterior_races.csv
#   {OUT}/posterior_coefs.csv
#   {OUT}/dcsi_race.csv                 (TRAIN predictions)
#
# Outputs (TEST/VAL predict):
#   {OUT}/dcsi_race.csv                 (heldout predictions using TRAIN posterior)

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# ---------------- Paths & config ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_parent_model")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
FIG_DIR = os.path.join(OUT_DIR, "figs"); Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

SPLITS_CSV   = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
SPLIT_TARGET = os.environ.get("F1_SPLIT_TARGET", "TRAIN").upper()
RUN_MODE_ENV = os.environ.get("F1_RUN_MODE", "").lower()   # "" => auto

INFER_METHOD = os.environ.get("F1_INFER", "advi").lower()
ADVI_STEPS   = int(os.environ.get("F1_ADVI_STEPS", "6000"))
NUTS_DRAWS   = int(os.environ.get("F1_NUTS_DRAWS", "1000"))
NUTS_TUNE    = int(os.environ.get("F1_NUTS_TUNE", "1000"))
RNG_SEED     = int(os.environ.get("F1_SEED", "123"))

PRIOR_SD_ALPHA = 0.50
PRIOR_SD_BETA  = 0.25
PRIOR_SD_D     = 0.30
PRIOR_SD_T     = 0.25
LIKELIHOOD_SD_INIT = 0.20

# ---------------- Load data ----------------
entries = pd.read_csv(os.path.join(DATA_DIR, "race_entries.csv"))
drivers = pd.read_csv(os.path.join(DATA_DIR, "drivers.csv"))
teams   = pd.read_csv(os.path.join(DATA_DIR, "constructors.csv"))
races   = pd.read_csv(os.path.join(DATA_DIR, "races.csv"))

# ensure date exists for ordering
if "date" not in races.columns:
    races = races.copy()
    races["date"] = pd.to_datetime(races["season_id"].astype(str) + "-01-01") + pd.to_timedelta(races["round"] * 14, unit="D")

# merge core fields
df = (entries
      .merge(drivers[["driver_id","driver_name"]], on="driver_id", how="left")
      .merge(teams[["constructor_id","constructor_name"]], on="constructor_id", how="left")
      .merge(races[["race_id","season_id","round","track_id","weather","date"]],
             on="race_id", how="left"))
df = df[(df["classified"] == True) & (~df["time_gap_s"].isna())].copy()

# perf target (higher=better)
eps = 1e-6
g = df.groupby("race_id")["time_gap_s"]
df["perf"] = 1.0 - (df["time_gap_s"] - g.transform("min")) / (g.transform("max") - g.transform("min") + eps)
df["perf"] = df["perf"].clip(0.0, 1.0)

# features
df["street"] = df["street"].astype(int) if "street" in df.columns else 0
df["wet"]    = (df["weather"].fillna("").str.lower().str.contains("wet|mixed")).astype(int)
df["grid_c"] = df["grid"] - df.groupby("race_id")["grid"].transform("mean")

# apply split
if not os.path.exists(SPLITS_CSV):
    raise FileNotFoundError(f"Missing splits file: {SPLITS_CSV}")
splits = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
df = df.merge(splits, on="race_id", how="left")

if SPLIT_TARGET not in {"TRAIN","TEST","VAL"}:
    raise ValueError("F1_SPLIT_TARGET must be TRAIN, TEST, or VAL")

# sort by time
df = df.sort_values(["season_id","round","date"]).reset_index(drop=True)

# decide run mode
RUN_MODE = RUN_MODE_ENV if RUN_MODE_ENV in {"fit","predict"} else ("fit" if SPLIT_TARGET=="TRAIN" else "predict")
print(f"[INFO] Split={SPLIT_TARGET} | Mode={RUN_MODE}")

# ---------------- Helpers ----------------
def build_index_map(series):
    uniq = series.astype("category").cat.categories.tolist()
    return {k:i for i,k in enumerate(uniq)}, uniq

def write_posterior_tables(idata, drv_levels, tm_levels, race_levels, feat_names):
    post = idata.posterior

    # drivers
    drv_mean = post["theta_d"].mean(dim=["chain","draw"]).values
    drv_q3   = np.percentile(post["theta_d"].values, 3, axis=(0,1))
    drv_q97  = np.percentile(post["theta_d"].values, 97, axis=(0,1))
    drv_tbl = pd.DataFrame({
        "driver_id": drv_levels,
        "mean": drv_mean,
        "hdi_3%": drv_q3,
        "hdi_97%": drv_q97
    })
    drv_tbl.to_csv(os.path.join(OUT_DIR, "posterior_drivers.csv"), index=False)

    # constructors
    tm_mean = post["theta_t"].mean(dim=["chain","draw"]).values
    tm_q3   = np.percentile(post["theta_t"].values, 3, axis=(0,1))
    tm_q97  = np.percentile(post["theta_t"].values, 97, axis=(0,1))
    tm_tbl = pd.DataFrame({
        "constructor_id": tm_levels,
        "mean": tm_mean,
        "hdi_3%": tm_q3,
        "hdi_97%": tm_q97
    })
    tm_tbl.to_csv(os.path.join(OUT_DIR, "posterior_constructors.csv"), index=False)

    # (optional) race effects if modeled (not in this minimalist spec); leave placeholder
    pd.DataFrame({"race_id": race_levels}).to_csv(os.path.join(OUT_DIR, "posterior_races.csv"), index=False)

    # fixed effects
    beta_mean = post["beta"].mean(dim=["chain","draw"]).values
    beta_q3   = np.percentile(post["beta"].values, 3, axis=(0,1))
    beta_q97  = np.percentile(post["beta"].values, 97, axis=(0,1))
    rows = [{"name":"alpha",
             "mean": float(post["alpha"].mean().values),
             "hdi_3%": float(np.percentile(post["alpha"].values, 3)),
             "hdi_97%": float(np.percentile(post["alpha"].values,97))}]
    for i,nm in enumerate(feat_names):
        rows.append({"name": nm, "mean": beta_mean[i], "hdi_3%": beta_q3[i], "hdi_97%": beta_q97[i]})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "posterior_coefs.csv"), index=False)

def dump_predictions(df_sub, drv_map_mean, tm_map_mean, alpha_mean, beta_mean, fname):
    """Write dcsi_race.csv-compatible predictions for the given df subset."""
    X = df_sub[["grid_c","street","wet"]].to_numpy(float)
    drv = df_sub["driver_id"].map(drv_map_mean).fillna(0.0).to_numpy(float)
    tm  = df_sub["constructor_id"].map(tm_map_mean).fillna(0.0).to_numpy(float)
    pred = float(alpha_mean) + drv + tm + (X @ beta_mean)

    out = df_sub[[
        "race_id","season_id","round",
        "driver_id","driver_name","constructor_id","constructor_name",
        "grid","finish_position","perf","street","wet"
    ]].copy()
    out["pred_perf"] = pred
    # expose the components used
    out["driver_eff_mean"] = drv
    out["team_eff_mean"]   = tm
    out.sort_values(["season_id","round","driver_name"], inplace=True)
    out.to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f"✅ wrote {os.path.join(OUT_DIR, fname)}")

# ---------------- PREDICT path (TEST/VAL) ----------------
posterior_nc = os.path.join(OUT_DIR, "posterior.nc")
if RUN_MODE == "predict":
    if not os.path.exists(posterior_nc):
        raise FileNotFoundError(f"Need TRAIN posterior to predict: {posterior_nc}")

    idata = az.from_netcdf(posterior_nc)
    post  = idata.posterior

    # posterior means
    alpha_mean = float(post["alpha"].mean().values)
    beta_mean  = post["beta"].mean(dim=["chain","draw"]).values

    # load driver/constructor level order used at TRAIN fit time
    drv_tbl = pd.read_csv(os.path.join(OUT_DIR, "posterior_drivers.csv"))
    tm_tbl  = pd.read_csv(os.path.join(OUT_DIR, "posterior_constructors.csv"))
    drv_map_mean = dict(zip(drv_tbl["driver_id"], drv_tbl["mean"]))
    tm_map_mean  = dict(zip(tm_tbl["constructor_id"], tm_tbl["mean"]))

    df_holdout = df[df["split"] == SPLIT_TARGET].copy()
    if df_holdout.empty:
        raise RuntimeError(f"No rows found for split={SPLIT_TARGET} in dataset.")

    dump_predictions(df_holdout, drv_map_mean, tm_map_mean, alpha_mean, beta_mean, "dcsi_race.csv")
    raise SystemExit(0)

# ---------------- FIT path (TRAIN) ----------------
df_train = df[df["split"] == "TRAIN"].copy()
if df_train.empty:
    raise RuntimeError("TRAIN split has no rows. Check splits.csv and data.")

# indexers on TRAIN universe
drv_index, drv_levels = build_index_map(df_train["driver_id"])
tm_index,  tm_levels  = build_index_map(df_train["constructor_id"])
race_index, race_levels = build_index_map(df_train["race_id"])

di = df_train["driver_id"].map(drv_index).to_numpy()
ti = df_train["constructor_id"].map(tm_index).to_numpy()
# features
X = df_train[["grid_c","street","wet"]].to_numpy(float)
y = df_train["perf"].to_numpy(float)

with pm.Model() as model:
    alpha = pm.Normal("alpha", mu=0.0, sigma=PRIOR_SD_ALPHA)
    beta  = pm.Normal("beta",  mu=0.0, sigma=PRIOR_SD_BETA, shape=X.shape[1])

    theta_d = pm.Normal("theta_d", mu=0.0, sigma=PRIOR_SD_D, shape=len(drv_levels))
    theta_t = pm.Normal("theta_t", mu=0.0, sigma=PRIOR_SD_T, shape=len(tm_levels))

    sigma = pm.HalfNormal("sigma", sigma=LIKELIHOOD_SD_INIT)

    mu = alpha + theta_d[di] + theta_t[ti] + pm.math.dot(X, beta)
    pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

    if INFER_METHOD == "nuts":
        idata = pm.sample(draws=NUTS_DRAWS, tune=NUTS_TUNE, chains=2, target_accept=0.9,
                          random_seed=RNG_SEED, progressbar=False)
    else:
        approx = pm.fit(n=ADVI_STEPS, random_seed=RNG_SEED, progressbar=False)
        idata  = approx.sample(draws=1500, random_seed=RNG_SEED)

# save posterior
az.to_netcdf(idata, posterior_nc)
print(f"💾 saved posterior to {posterior_nc}")

# write posterior summaries
write_posterior_tables(idata, drv_levels, tm_levels, race_levels, ["grid_c","street","wet"])

# also produce TRAIN predictions for convenience/plots
post = idata.posterior
alpha_mean = float(post["alpha"].mean().values)
beta_mean  = post["beta"].mean(dim=["chain","draw"]).values

drv_tbl = pd.read_csv(os.path.join(OUT_DIR, "posterior_drivers.csv"))
tm_tbl  = pd.read_csv(os.path.join(OUT_DIR, "posterior_constructors.csv"))
drv_map_mean = dict(zip(drv_tbl["driver_id"], drv_tbl["mean"]))
tm_map_mean  = dict(zip(tm_tbl["constructor_id"], tm_tbl["mean"]))

dump_predictions(df_train, drv_map_mean, tm_map_mean, alpha_mean, beta_mean, "dcsi_race.csv")