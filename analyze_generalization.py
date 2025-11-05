import json
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ---------- 1. Load TRAIN and TEST metrics ----------

train_path = Path("outputs/f1_dynamic_train_pit_sc/prob_metrics.json")
test_path  = Path("outputs/f1_dynamic_test_pit_sc/prob_metrics.json")

with open(train_path, "r") as f:
    train = json.load(f)
with open(test_path, "r") as f:
    test = json.load(f)

# (category, metric) pairs matching the JSON structure
metric_specs = [
    ("win",    "brier"),
    ("podium", "brier"),
    ("points", "brier"),
    ("win",    "logloss"),
    ("podium", "logloss"),
    ("points", "logloss"),
]

print("Δ = TEST - TRAIN\n")

labels = []
diffs = []

for cat, met in metric_specs:
    t_train = train[cat][met]
    t_test  = test[cat][met]
    diff = t_test - t_train
    label = f"{cat.capitalize()} {met.capitalize()}"
    labels.append(label)
    diffs.append(diff)
    print(f"{label:18s} {t_train:.4f} → {t_test:.4f}  (Δ {diff:+.4f})")

# also show tau
tau_train = train.get("tau_opt", None)
tau_test  = test.get("tau_opt", None)
if tau_train is not None and tau_test is not None:
    print(f"\nTau (temperature): TRAIN {tau_train:.3f} vs TEST {tau_test:.3f}")

# ---------- 2. Plot calibration differences ----------

plt.figure()
colors = ["#0072B2" if d < 0 else "#D55E00" for d in diffs]
plt.barh(labels, diffs, color=colors)
plt.axvline(0, color="k", lw=1)
plt.title("Calibration Difference (TEST − TRAIN)")
plt.xlabel("Δ (lower = better generalization)")
plt.tight_layout()
Path("outputs/f1_dynamic_test_pit_sc").mkdir(parents=True, exist_ok=True)
plt.savefig("outputs/f1_dynamic_test_pit_sc/calibration_diff.png")
plt.show()

# ---------- 3. Driver & team effect robustness ----------

# You can comment this section out if the CSVs don't exist yet.

drv_train_path = Path("outputs/f1_dynamic_train_pit_sc/dcsi_cumulative_drivers.csv")
drv_test_path  = Path("outputs/f1_dynamic_test_pit_sc/dcsi_cumulative_drivers.csv")
tm_train_path  = Path("outputs/f1_dynamic_train_pit_sc/dcsi_cumulative_constructors.csv")
tm_test_path   = Path("outputs/f1_dynamic_test_pit_sc/dcsi_cumulative_constructors.csv")

if drv_train_path.exists() and drv_test_path.exists():
    drv_train = pd.read_csv(drv_train_path)
    drv_test  = pd.read_csv(drv_test_path)

    drv_merge = drv_train.merge(drv_test, on="driver_id", suffixes=("_train", "_test"))

    # Adjust these column names if your CSV uses slightly different ones
    eff_train_col = "driver_eff_mean_train"
    eff_test_col  = "driver_eff_mean_test"

    if eff_train_col not in drv_merge.columns or eff_test_col not in drv_merge.columns:
        # try fallback names (without suffixes)
        eff_train_col = "driver_eff_mean_train" if "driver_eff_mean_train" in drv_merge.columns else "driver_eff_mean_x"
        eff_test_col  = "driver_eff_mean_test" if "driver_eff_mean_test" in drv_merge.columns else "driver_eff_mean_y"

    drv_merge["delta_effect"] = drv_merge[eff_test_col] - drv_merge[eff_train_col]

    name_col = "driver_name_train" if "driver_name_train" in drv_merge.columns else "driver_name_x"

    print("\nDrivers with biggest performance drop (TEST − TRAIN):")
    print(
        drv_merge.sort_values("delta_effect")
        [[name_col, "delta_effect"]]
        .head(10)
        .to_string(index=False)
    )

    # scatter plot for driver robustness
    plt.figure()
    plt.scatter(drv_merge[eff_train_col], drv_merge[eff_test_col], alpha=0.7)
    plt.axline((0, 0), (1, 1), color="k", linestyle="--")
    plt.xlabel("Train Effect")
    plt.ylabel("Test Effect")
    plt.title("Driver Effect Generalization (Pit+SC Model)")
    plt.tight_layout()
    plt.savefig("outputs/f1_dynamic_test_pit_sc/driver_effect_generalization.png")
    plt.show()

if tm_train_path.exists() and tm_test_path.exists():
    tm_train = pd.read_csv(tm_train_path)
    tm_test  = pd.read_csv(tm_test_path)

    tm_merge = tm_train.merge(tm_test, on="constructor_id", suffixes=("_train", "_test"))

    teff_train_col = "team_eff_mean_train"
    teff_test_col  = "team_eff_mean_test"

    if teff_train_col not in tm_merge.columns or teff_test_col not in tm_merge.columns:
        teff_train_col = "team_eff_mean_train" if "team_eff_mean_train" in tm_merge.columns else "team_eff_mean_x"
        teff_test_col  = "team_eff_mean_test" if "team_eff_mean_test" in tm_merge.columns else "team_eff_mean_y"

    tm_merge["delta_effect"] = tm_merge[teff_test_col] - tm_merge[teff_train_col]

    tname_col = "constructor_name_train" if "constructor_name_train" in tm_merge.columns else "constructor_name_x"

    print("\nTeams with biggest performance drop (TEST − TRAIN):")
    print(
        tm_merge.sort_values("delta_effect")
        [[tname_col, "delta_effect"]]
        .head(10)
        .to_string(index=False)
    )
