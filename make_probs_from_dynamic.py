# make_probs_from_dynamic.py
# Convert dynamic pred_perf to calibrated win / podium / points probabilities
# using a Plackett–Luce approximation (Gumbel-top-k simulation) + temperature calibration.
#
# Inputs:
#   - outputs/f1_dynamic/dcsi_race.csv (from f1_dynamic_update.py)
#   - data/.../race_entries.csv (to read synthetic implied odds/probs if present)
# Env:
#   F1_DATA_DIR (default: data/synth_f1_2018_2025_realish)
#   F1_OUT_DIR  (default: outputs/f1_dynamic)
#
# Outputs:
#   - outputs/f1_dynamic/dcsi_probs.csv
#   - outputs/f1_dynamic/figs/prob_calibration_{win,podium,points}.png
#   - outputs/f1_dynamic/prob_metrics.json

import os, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
DATA_DIR = os.environ.get("F1_DATA_DIR", "data/synth_f1_2018_2025_realish")
OUT_DIR  = os.environ.get("F1_OUT_DIR",  "outputs/f1_dynamic")
FIG_DIR  = os.path.join(OUT_DIR, "figs")
Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

DCSI_RACE = os.path.join(OUT_DIR, "dcsi_race.csv")
ENTRIES   = os.path.join(DATA_DIR, "race_entries.csv")

# ---------------- Load ----------------
if not os.path.exists(DCSI_RACE):
    raise FileNotFoundError(f"Missing {DCSI_RACE}. Run f1_dynamic_update.py first.")

r = pd.read_csv(DCSI_RACE)
entries = pd.read_csv(ENTRIES) if os.path.exists(ENTRIES) else pd.DataFrame()

# binary outcomes for calibration
r["win"]    = (r.groupby("race_id")["finish_position"].transform("min") == r["finish_position"]).astype(int)
r["podium"] = (r["finish_position"] <= 3).astype(int)
r["points"] = (r["finish_position"] <= 10).astype(int)

# Optional merge synthetic implied probs for diagnostics
if "implied_p_post" in entries.columns:
    r = r.merge(entries[["race_id","driver_id","implied_p_post","implied_p_pre"]],
                on=["race_id","driver_id"], how="left")

# ---------------- Helpers ----------------
def softmax(x, tau=1.0):
    z = (x - np.max(x)) / max(tau, 1e-6)
    e = np.exp(z)
    return e / e.sum()

def brier(y_true, p):
    y = np.asarray(y_true, float)
    p = np.asarray(p, float)
    return float(np.mean((y - p) ** 2))

def logloss(y_true, p, eps=1e-12):
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def calibrate_tau(df, grid=np.linspace(0.35, 1.50, 24)):
    """Choose temperature tau minimizing average Brier for p(win)."""
    best_tau, best_brier = None, 1e9
    for tau in grid:
        preds = []
        trues = []
        for _, g in df.groupby("race_id"):
            p = softmax(g["pred_perf"].to_numpy(), tau=tau)
            # align to g order
            preds.extend(p.tolist())
            # one-win-per-race indicator already in g["win"]
            trues.extend(g["win"].to_numpy().tolist())
        score = brier(trues, preds)
        if score < best_brier:
            best_brier, best_tau = score, tau
    return float(best_tau), float(best_brier)

def gumbel_topk_probs(scores, draws=500, rng=None):
    """
    Plackett–Luce approximation via Gumbel-Max trick.
    Returns marginal probabilities for:
      - win (rank 1)
      - podium (rank <=3)
      - points (rank <=10)
    scores: np.array of utilities (higher better)
    """
    if rng is None:
        rng = np.random.default_rng(123)
    n = len(scores)
    wins = np.zeros(n, dtype=float)
    pod  = np.zeros(n, dtype=float)
    pts  = np.zeros(n, dtype=float)
    for _ in range(draws):
        noise = rng.gumbel(size=n)
        rank = np.argsort(-(scores + noise))  # descending
        wins[rank[0]] += 1.0
        pod[rank[:3]] += 1.0
        pts[rank[:10]] += 1.0
    return wins / draws, pod / draws, pts / draws

# ---------------- Temperature calibration ----------------
tau_opt, brier_win = calibrate_tau(r)
print(f"Calibrated temperature tau={tau_opt:.3f} (Brier(win)={brier_win:.4f})")

# ---------------- Compute probabilities per race ----------------
rows = []
rng = np.random.default_rng(1234)
for rid, g in r.groupby("race_id"):
    g = g.copy().reset_index(drop=True)
    s = g["pred_perf"].to_numpy()
    # temperature scaling
    s_scaled = s / max(tau_opt, 1e-6)

    # Plackett–Luce approximation
    p_win, p_podium, p_points = gumbel_topk_probs(s_scaled, draws=600, rng=rng)

    # Optional blend with synthetic implied probs (post-quali) for diagnostics only
    if "implied_p_post" in g.columns:
        # Light blending (10%) to avoid overfitting to synthetic odds, can tune or disable
        blend_w = 0.10
        ip = g["implied_p_post"].fillna(0.0).to_numpy()
        ip = ip / max(ip.sum(), 1e-9)  # renormalize
        p_win = (1 - blend_w) * p_win + blend_w * ip

    out = g[["race_id","season_id","round","driver_id","driver_name","constructor_id","constructor_name",
             "finish_position"]].copy()
    out["p_win"]    = p_win
    out["p_podium"] = p_podium
    out["p_points"] = p_points
    rows.append(out)

probs = pd.concat(rows, ignore_index=True)

# ---------------- Metrics & reliability plots ----------------
def reliability_plot(df, pcol, ycol, title, path):
    # decile bins
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
    return {
        "brier": brier(df[ycol], df[pcol]),
        "logloss": logloss(df[ycol], df[pcol]),
        "n": int(len(df))
    }

metrics = {
    "win":    reliability_plot(probs.merge(r[["race_id","driver_id","win"]], on=["race_id","driver_id"]),
                               "p_win", "win",
                               "Calibration: Win", os.path.join(FIG_DIR, "prob_calibration_win.png")),
    "podium": reliability_plot(probs.merge(r[["race_id","driver_id","podium"]], on=["race_id","driver_id"]),
                               "p_podium", "podium",
                               "Calibration: Podium (Top 3)", os.path.join(FIG_DIR, "prob_calibration_podium.png")),
    "points": reliability_plot(probs.merge(r[["race_id","driver_id","points"]], on=["race_id","driver_id"]),
                               "p_points", "points",
                               "Calibration: Points (Top 10)", os.path.join(FIG_DIR, "prob_calibration_points.png")),
    "tau_opt": tau_opt
}

# ---------------- Save ----------------
out_path = os.path.join(OUT_DIR, "dcsi_probs.csv")
probs.to_csv(out_path, index=False)
with open(os.path.join(OUT_DIR, "prob_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Wrote probabilities to:", out_path)
print("📈 Saved calibration figures to:", FIG_DIR)
print("📊 Metrics:", json.dumps(metrics, indent=2))
