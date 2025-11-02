# make_probs_from_dynamic.py
# Convert pred_perf -> calibrated probabilities (win/podium/points)
# Split-aware + frozen temperature support (τ from TRAIN).
#
# Env:
#   F1_DATA_DIR
#   F1_OUT_DIR
#   F1_SPLITS_CSV
#   F1_SPLIT_TARGET=TRAIN|TEST|VAL
#   F1_TAU_PATH=/path/to/train/prob_metrics.json  (optional, recommended for TEST)
#   F1_TAU_VALUE=0.97                             (optional numeric override)

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths & split handling ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic")
FIG_DIR  = os.path.join(OUT_DIR, "figs")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

SPLITS_CSV   = os.environ.get("F1_SPLITS_CSV", "outputs/splits/splits.csv")
SPLIT_ENV    = os.environ.get("F1_SPLIT_TARGET", "").upper()

def infer_split_from_outdir(path: str):
    lower = path.lower()
    if "train" in lower: return "TRAIN"
    if "test"  in lower: return "TEST"
    if "val"   in lower: return "VAL"
    return ""

SPLIT_TARGET = SPLIT_ENV or infer_split_from_outdir(OUT_DIR)

DCSI_RACE = os.path.join(OUT_DIR, "dcsi_race.csv")
ENTRIES   = os.path.join(DATA_DIR, "race_entries.csv")

if not os.path.exists(DCSI_RACE):
    raise FileNotFoundError(f"Missing {DCSI_RACE}. Run f1_dynamic_update.py for this split first.")

r = pd.read_csv(DCSI_RACE)
entries = pd.read_csv(ENTRIES) if os.path.exists(ENTRIES) else pd.DataFrame()

# If splits.csv exists, enforce the split selection (and never include VAL unless explicitly chosen)
if os.path.exists(SPLITS_CSV) and "race_id" in r.columns:
    s = pd.read_csv(SPLITS_CSV)[["race_id","split"]]
    r = r.merge(s, on="race_id", how="left")
    if SPLIT_TARGET in {"TRAIN","TEST","VAL"}:
        r = r[r["split"] == SPLIT_TARGET]
    else:
        r = r[r["split"].isin(["TRAIN","TEST"])]

# ---------------- Outcomes for calibration ----------------
r["win"]    = (r.groupby("race_id")["finish_position"].transform("min") == r["finish_position"]).astype(int)
r["podium"] = (r["finish_position"] <= 3).astype(int)
r["points"] = (r["finish_position"] <= 10).astype(int)

# Optional implied odds
if "implied_p_post" in entries.columns:
    r = r.merge(entries[["race_id","driver_id","implied_p_post","implied_p_pre"]],
                on=["race_id","driver_id"], how="left")

# ---------------- Helpers ----------------
def softmax(x, tau=1.0):
    z = (x - np.max(x)) / max(tau, 1e-6)
    e = np.exp(z)
    return e / e.sum()

def brier(y_true, p):
    y = np.asarray(y_true, float); p = np.asarray(p, float)
    return float(np.mean((y - p) ** 2))

def logloss(y_true, p, eps=1e-12):
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def calibrate_tau(df, grid=np.linspace(0.35, 1.50, 24)):
    best_tau, best_brier = None, 1e9
    for tau in grid:
        preds, trues = [], []
        for _, g in df.groupby("race_id", observed=True):
            p = softmax(g["pred_perf"].to_numpy(), tau=tau)
            preds.extend(p.tolist())
            trues.extend(g["win"].to_numpy().tolist())
        score = brier(trues, preds)
        if score < best_brier:
            best_tau, best_brier = tau, score
    return float(best_tau), float(best_brier)

def gumbel_topk_probs(scores, draws=600, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    n = len(scores)
    wins = np.zeros(n); pod = np.zeros(n); pts = np.zeros(n)
    for _ in range(draws):
        rank = np.argsort(-(scores + rng.gumbel(size=n)))
        wins[rank[0]] += 1
        pod[rank[:3]] += 1
        pts[rank[:10]] += 1
    return wins/draws, pod/draws, pts/draws

# ---------------- Get τ (frozen if available) ----------------
TAU_VALUE_ENV = os.environ.get("F1_TAU_VALUE")
TAU_PATH      = os.environ.get("F1_TAU_PATH")  # expected to point to TRAIN prob_metrics.json

tau_opt = None
if TAU_VALUE_ENV:
    try:
        tau_opt = float(TAU_VALUE_ENV)
        print(f"[INFO] Using τ from F1_TAU_VALUE = {tau_opt:.3f}")
    except:
        pass

if tau_opt is None and TAU_PATH and os.path.exists(TAU_PATH):
    try:
        with open(TAU_PATH, "r") as f:
            m = json.load(f)
        tau_opt = float(m.get("tau_opt", m.get("tau_opt_softmax")))
        print(f"[INFO] Using τ from TRAIN metrics @ {TAU_PATH} = {tau_opt:.3f}")
    except Exception as e:
        print(f"[WARN] Failed to read τ from {TAU_PATH}: {e}")

# Auto-discover sibling TRAIN folder if not provided
if tau_opt is None:
    guess = OUT_DIR.replace("_test", "_train")
    candidate = os.path.join(guess, "prob_metrics.json")
    if os.path.exists(candidate):
        try:
            with open(candidate, "r") as f:
                m = json.load(f)
            tau_opt = float(m.get("tau_opt", m.get("tau_opt_softmax")))
            print(f"[INFO] Using τ from TRAIN metrics @ {candidate} = {tau_opt:.3f}")
        except Exception as e:
            print(f"[WARN] Failed to read τ from {candidate}: {e}")

# If still None, calibrate (but warn if split is TEST/VAL)
if tau_opt is None:
    if SPLIT_TARGET in {"TEST","VAL"}:
        print("[WARN] τ not provided; calibrating on non-TRAIN split (risk of leakage). "
              "Set F1_TAU_PATH to TRAIN prob_metrics.json to avoid this.")
    tau_opt, brier_win = calibrate_tau(r)
    print(f"[{SPLIT_TARGET or 'DEFAULT'}] Calibrated τ={tau_opt:.3f} (Brier win={brier_win:.4f})")
else:
    print(f"[{SPLIT_TARGET or 'DEFAULT'}] Frozen τ={tau_opt:.3f} (from TRAIN)")

# ---------------- Compute probabilities per race ----------------
rows = []
rng = np.random.default_rng(1234)
for rid, g in r.groupby("race_id", observed=True):
    g = g.copy().reset_index(drop=True)
    s = g["pred_perf"].to_numpy() / max(tau_opt, 1e-6)

    p_win, p_podium, p_points = gumbel_topk_probs(s, draws=600, rng=rng)

    # light blend with implied post-quali probs if available (diagnostic only)
    if "implied_p_post" in g.columns:
        w = 0.10
        ip = g["implied_p_post"].fillna(0.0).to_numpy()
        if ip.sum() > 0:
            ip = ip / ip.sum()
            p_win = (1 - w) * p_win + w * ip

    out = g[["race_id","season_id","round","driver_id","driver_name",
             "constructor_id","constructor_name","finish_position"]].copy()
    out["p_win"]    = p_win
    out["p_podium"] = p_podium
    out["p_points"] = p_points
    rows.append(out)

probs = pd.concat(rows, ignore_index=True)

# ---------------- Reliability plots & metrics ----------------
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
    plt.xlabel("Predicted probability bin mean")
    plt.ylabel("Empirical rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return {"brier": brier(df[ycol], df[pcol]),
            "logloss": logloss(df[ycol], df[pcol]),
            "n": int(len(df))}

metrics = {
    "split": SPLIT_TARGET or "UNSPECIFIED",
    "tau_opt": float(tau_opt),
    "win":    reliability_plot(probs.merge(r[["race_id","driver_id","win"]], on=["race_id","driver_id"]),
                               "p_win","win","Calibration: Win",
                               os.path.join(FIG_DIR, "prob_calibration_win.png")),
    "podium": reliability_plot(probs.merge(r[["race_id","driver_id","podium"]], on=["race_id","driver_id"]),
                               "p_podium","podium","Calibration: Podium",
                               os.path.join(FIG_DIR, "prob_calibration_podium.png")),
    "points": reliability_plot(probs.merge(r[["race_id","driver_id","points"]], on=["race_id","driver_id"]),
                               "p_points","points","Calibration: Points",
                               os.path.join(FIG_DIR, "prob_calibration_points.png")),
}

# ---------------- Save ----------------
out_path = os.path.join(OUT_DIR, "dcsi_probs.csv")
probs.to_csv(out_path, index=False)
with open(os.path.join(OUT_DIR, "prob_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Wrote probabilities to:", out_path)
print("📈 Saved calibration figures to:", FIG_DIR)
print("📊 Metrics:", json.dumps(metrics, indent=2))
