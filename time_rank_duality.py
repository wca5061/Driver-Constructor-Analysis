# time_rank_duality.py
# Time–Rank Duality evaluation with frozen τ (from TRAIN)
#
# Env:
#   F1_OUT_DIR, F1_SPLITS_CSV, F1_SPLIT_TARGET
#   F1_TAU_PATH=/path/to/train/prob_metrics.json
#   F1_TAU_VALUE=0.97

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic_test")
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
SPLIT_TARGET = SPLIT_ENV or infer_split_from_outdir(OUT_DIR)

DCSI_RACE    = os.path.join(OUT_DIR, "dcsi_race.csv")
if not os.path.exists(DCSI_RACE):
    raise FileNotFoundError(f"Missing {DCSI_RACE}. Run the pipeline to generate it first.")
r = pd.read_csv(DCSI_RACE)

if os.path.exists(SPLITS_CSV) and "race_id" in r.columns:
    s = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
    r = r.merge(s, on="race_id", how="left")
    if SPLIT_TARGET in {"TRAIN","TEST","VAL"}:
        r = r[r["split"] == SPLIT_TARGET]
    else:
        r = r[r["split"].isin(["TRAIN","TEST"])]
if r.empty:
    raise RuntimeError(f"No rows for split={SPLIT_TARGET}.")

# Outcomes
r["win"]    = (r.groupby("race_id")["finish_position"].transform("min") == r["finish_position"]).astype(int)
r["podium"] = (r["finish_position"] <= 3).astype(int)
r["points"] = (r["finish_position"] <= 10).astype(int)

# ---- τ handling (frozen preferred) ----
TAU_VALUE_ENV = os.environ.get("F1_TAU_VALUE")
TAU_PATH      = os.environ.get("F1_TAU_PATH")
tau_opt = None
if TAU_VALUE_ENV:
    try: tau_opt = float(TAU_VALUE_ENV); print(f"[INFO] Using τ from F1_TAU_VALUE = {tau_opt:.3f}")
    except: pass
if tau_opt is None and TAU_PATH and os.path.exists(TAU_PATH):
    try:
        with open(TAU_PATH, "r") as f: m = json.load(f)
        tau_opt = float(m.get("tau_opt", m.get("tau_opt_softmax")))
        print(f"[INFO] Using τ from TRAIN metrics @ {TAU_PATH} = {tau_opt:.3f}")
    except Exception as e:
        print(f"[WARN] Failed to read τ from {TAU_PATH}: {e}")
if tau_opt is None:
    guess = OUT_DIR.replace("_test", "_train")
    candidate = os.path.join(guess, "prob_metrics.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r") as f: m = json.load(f)
            tau_opt = float(m.get("tau_opt", m.get("tau_opt_softmax")))
            print(f"[INFO] Using τ from TRAIN metrics @ {candidate} = {tau_opt:.3f}")
        except Exception as e:
            print(f"[WARN] Failed to read τ from {candidate}: {e}")

def brier(y_true, p):
    y = np.asarray(y_true, float); p = np.asarray(p, float)
    return float(np.mean((y - p) ** 2))
def logloss(y_true, p, eps=1e-12):
    y = np.asarray(y_true, float); p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
def reliability_plot(df, pcol, ycol, title, path):
    q = pd.qcut(df[pcol], q=10, labels=False, duplicates="drop")
    grp = df.groupby(q, observed=True).agg(
        p_mean=(pcol, "mean"),
        y_rate=(ycol, "mean"),
        n=(ycol, "size")
    ).reset_index(drop=True)
    plt.figure(figsize=(5.2,4.2))
    plt.plot(grp["p_mean"], grp["y_rate"], marker="o")
    plt.plot([0,1],[0,1],"--",linewidth=1)
    plt.xlabel("Predicted probability bin mean"); plt.ylabel("Empirical rate")
    plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    return grp
def softmax(x, tau=1.0):
    z = (x - np.max(x)) / max(tau, 1e-6); e = np.exp(z); return e / e.sum()
def calibrate_tau_win(df, grid=np.linspace(0.35, 1.50, 24)):
    best_tau, best_brier = None, 1e9
    for tau in grid:
        preds, trues = [], []
        for _, g in df.groupby("race_id", observed=True):
            p = softmax(g["pred_perf"].to_numpy(), tau=tau)
            preds.extend(p.tolist())
            trues.extend(((g["finish_position"] == g["finish_position"].min()).astype(int)).to_numpy().tolist())
        score = brier(trues, preds)
        if score < best_brier: best_tau, best_brier = tau, score
    return float(best_tau), float(best_brier)
def gumbel_topk_probs(scores, draws=800, rng=None):
    if rng is None: rng = np.random.default_rng(2025)
    n = len(scores); wins = np.zeros(n); pod = np.zeros(n); pts = np.zeros(n)
    for _ in range(draws):
        rank = np.argsort(-(scores + rng.gumbel(size=n)))
        wins[rank[0]] += 1; pod[rank[:3]] += 1; pts[rank[:10]] += 1
    return wins/draws, pod/draws, pts/draws
def plackett_luce_topk_probs(scores, draws=800, rng=None):
    if rng is None: rng = np.random.default_rng(340)
    n = len(scores); w = np.exp(scores - np.max(scores))
    wins = np.zeros(n); pod = np.zeros(n); pts = np.zeros(n)
    for _ in range(draws):
        alive = np.arange(n); w_alive = w.copy(); order = []
        for _k in range(n):
            p = w_alive / w_alive.sum()
            i = rng.choice(len(alive), p=p)
            choice = alive[i]; order.append(choice)
            alive = np.delete(alive, i); w_alive = np.delete(w_alive, i)
        order = np.array(order)
        wins[order[0]] += 1; pod[order[:3]] += 1; pts[order[:10]] += 1
    return wins/draws, pod/draws, pts/draws

# If τ absent and split is TEST/VAL, warn and calibrate (last resort)
if tau_opt is None:
    if SPLIT_TARGET in {"TEST","VAL"}:
        print("[WARN] τ not provided; calibrating on non-TRAIN split (risk of leakage). "
              "Set F1_TAU_PATH to TRAIN prob_metrics.json to avoid this.")
    tau_opt, _ = calibrate_tau_win(r)
    print(f"[{SPLIT_TARGET or 'DEFAULT'}] Calibrated τ={tau_opt:.3f}")
else:
    print(f"[{SPLIT_TARGET or 'DEFAULT'}] Frozen τ={tau_opt:.3f} (from TRAIN)")

# Compute both methods per race
rows = []
rng_time = np.random.default_rng(1234)
rng_rank = np.random.default_rng(5678)

for rid, g in r.groupby("race_id", observed=True):
    g = g.copy().reset_index(drop=True)
    s_time = g["pred_perf"].to_numpy() / max(tau_opt, 1e-6)
    p_win_time, p_pod_time, p_pts_time = gumbel_topk_probs(s_time, draws=900, rng=rng_time)

    s_rank = g["pred_perf"].to_numpy()
    p_win_pl, p_pod_pl, p_pts_pl = plackett_luce_topk_probs(s_rank, draws=900, rng=rng_rank)

    out = g[["race_id","season_id","round","driver_id","driver_name","constructor_id","constructor_name",
             "finish_position"]].copy()
    out["p_win_time"]    = p_win_time
    out["p_podium_time"] = p_pod_time
    out["p_points_time"] = p_pts_time
    out["p_win_pl"]      = p_win_pl
    out["p_podium_pl"]   = p_pod_pl
    out["p_points_pl"]   = p_pts_pl
    rows.append(out)

probs = pd.concat(rows, ignore_index=True).sort_values(["season_id","round","driver_name"])

def eval_block(df, pcol, ycol, title, fig_stub):
    m = {
        "brier":   brier(df[ycol], df[pcol]),
        "logloss": logloss(df[ycol], df[pcol]),
        "mean_p":  float(df[pcol].mean()),
        "mean_y":  float(df[ycol].mean()),
        "n":       int(len(df))
    }
    reliability_plot(df, pcol, ycol, f"{title}: {pcol}", os.path.join(FIG_DIR, f"{fig_stub}_{pcol}.png"))
    return m

metrics = {
    "split": SPLIT_TARGET or "UNSPECIFIED",
    "tau_opt_softmax": float(tau_opt),
    "time": {
        "win":    eval_block(probs.merge(r[["race_id","driver_id","win"]], on=["race_id","driver_id"]), "p_win_time",    "win",    "Calibration", "duality_cal_win"),
        "podium": eval_block(probs.merge(r[["race_id","driver_id","podium"]], on=["race_id","driver_id"]), "p_podium_time", "podium", "Calibration", "duality_cal_podium"),
        "points": eval_block(probs.merge(r[["race_id","driver_id","points"]], on=["race_id","driver_id"]), "p_points_time", "points", "Calibration", "duality_cal_points"),
    },
    "pl": {
        "win":    eval_block(probs.merge(r[["race_id","driver_id","win"]], on=["race_id","driver_id"]), "p_win_pl",    "win",    "Calibration", "duality_cal_win_pl"),
        "podium": eval_block(probs.merge(r[["race_id","driver_id","podium"]], on=["race_id","driver_id"]), "p_podium_pl", "podium", "Calibration", "duality_cal_podium_pl"),
        "points": eval_block(probs.merge(r[["race_id","driver_id","points"]], on=["race_id","driver_id"]), "p_points_pl", "points", "Calibration", "duality_cal_points_pl"),
    }
}

plt.figure(figsize=(5.2,4.2))
plt.scatter(probs["p_win_time"], probs["p_win_pl"], s=10, alpha=0.5)
plt.plot([0,1],[0,1],"--", linewidth=1)
plt.xlabel("Time-side p(win)"); plt.ylabel("PL p(win)")
plt.title(f"{SPLIT_TARGET}: Time vs Rank — Win Prob")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "duality_scatter_win_softmax_vs_pl.png"), dpi=150)
plt.close()

out_csv = os.path.join(OUT_DIR, "duality_probs.csv")
probs.to_csv(out_csv, index=False)
with open(os.path.join(OUT_DIR, "duality_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Wrote:", out_csv)
print("📊 Metrics:", json.dumps(metrics, indent=2))
print("🎨 Figures in:", FIG_DIR)
