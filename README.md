# 🏎️ Formula 1 Driver–Constructor Separation Index (DCSI)

This repository implements a **Dynamic Bayesian Driver–Constructor Separation Index (DCSI)** for Formula 1 — estimating how much of a team's performance comes from the driver vs. the car.  

The pipeline builds synthetic (real-ish) data, performs 70/20 train-test splits, fits a dynamic hierarchical model, evaluates calibration and time–rank duality, and exports a Streamlit dashboard.

---

## ⚙️ 0) Setup

### Requirements
```bash
# from repo root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt    # if present
# otherwise:
pip install numpy pandas matplotlib arviz pymc scipy fastf1 plotly streamlit
```

### Default Paths

| Purpose | Default Path |
|---------|--------------|
| Data | `data/synth_f1_2018_2025_realish/` |
| Splits | `outputs/splits/splits.csv` |
| Dynamic TRAIN | `outputs/f1_dynamic_train/` |
| Dynamic TEST | `outputs/f1_dynamic_test/` |
| Parent Model | `outputs/f1_parent_model/` |
| Dashboard JSON | `outputs/dashboard/dashboard_export.json` |

**No need to create these folders** — the scripts will.

---

## 🚀 1) Build the Synthetic Dataset
```bash
python dataset_builder.py
```

Creates CSVs under `data/synth_f1_2018_2025_realish/`  
(races, race_entries, drivers, constructors, laps, pitstops, safetycars, odds_snapshots, …)

---

## 🧩 2) Create 70/20 Train–Test Splits
```bash
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs" \
python create_splits.py
```

Produces → `outputs/splits/splits.csv`

---

## 🧠 3) Run Dynamic Model on TRAIN (Fits Priors)
```bash
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python f1_dynamic_update.py
```

**Outputs:**
- `dcsi_race.csv`, `dcsi_cumulative_drivers.csv`, `dcsi_cumulative_constructors.csv`
- `priors.npz`
- Plots in `figs/`

**Quick sanity check:**
```bash
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python quick_check_dynamic.py
```

---

## 🔮 4) Predict on TEST (Using TRAIN Priors)
```bash
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
F1_PRIORS_PATH="outputs/f1_dynamic_train/priors.npz" \
python f1_dynamic_update.py
```

Outputs → `outputs/f1_dynamic_test/dcsi_race.csv`

---

## 📈 5) Convert Scores → Probabilities (Frozen τ)

**Compute τ on TRAIN:**
```bash
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python make_probs_from_dynamic.py
```

**Apply τ on TEST:**
```bash
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
F1_TAU_PATH="outputs/f1_dynamic_train/prob_metrics.json" \
python make_probs_from_dynamic.py
```

**Outputs:**
- `dcsi_probs.csv`
- `prob_metrics.json`
- Calibration plots in `figs/`

---

## 🔍 6) Build Explainable Roll-Ups (TEST)
```bash
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
python build_rollups_explainable.py
```

**Outputs:**
- `rollup_drivers_overall.csv`, `rollup_teams_overall.csv`, etc.
- Figures in `figs/`

**Optional:**
```bash
python quick_check_dynamic.py
```

---

## ⏱️ 7) Time–Rank Duality Evaluation (TEST)
```bash
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
F1_TAU_PATH="outputs/f1_dynamic_train/prob_metrics.json" \
python time_rank_duality.py
```

**Outputs:**  
`duality_probs.csv`, `duality_metrics.json`, and calibration plots.

---

## 🧮 8) (Optional) Parent Model Path

**TRAIN:**
```bash
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_parent_model" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python f1_parent_model.py
```

**TEST:**
```bash
F1_SPLIT_TARGET=TEST \
python f1_parent_model.py
```

**Diagnostics:**
```bash
python f1_diagnostics.py
```

---

## 🧪 9) Sensitivity Sweep (TRAIN Only)
```bash
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
python sensitivity_sweep.py
```

Outputs → `outputs/f1_dynamic_train/sensitivity/summary.csv`

---

## ⚡ 10) Add Pit-Stop & Safety-Car Regressors (Phase II)

Enable in environment before running:
```bash
export F1_USE_PIT=1
export F1_USE_SC=1
```

Then rerun `f1_dynamic_update.py` for TRAIN and TEST, followed by:
```bash
python make_probs_from_dynamic.py
python build_rollups_explainable.py
python time_rank_duality.py
```

Outputs will be in `outputs/f1_dynamic_train_pit_sc/` and `outputs/f1_dynamic_test_pit_sc/`

---

## 🧭 11) Cross-Split Generalization Analysis
```bash
python analyze_generalization.py
```

Compares Brier & Log-loss differences (TEST – TRAIN) and plots driver/team robustness.

---

## 🗂️ 12) Export Dashboard JSON
```bash
python export_dashboard_json.py
```

Creates → `outputs/dashboard/dashboard_export.json`

---

## 🖥️ 13) Launch Streamlit Dashboard
```bash
streamlit run dashboard_app.py
```

**The dashboard shows:**
- Top 10 Outperforming Drivers
- Constructor Strength Rankings
- Driver & Team DCSI Trends
- Calibration & Time–Rank Duality plots

---

## ✅ Expected Outputs Checklist

| Folder | Key Files |
|--------|-----------|
| `outputs/splits/` | `splits.csv` |
| `outputs/f1_dynamic_train/` | `dcsi_race.csv`, `priors.npz`, `prob_metrics.json` |
| `outputs/f1_dynamic_test/` | `dcsi_probs.csv`, rollups, figs |
| `outputs/f1_dynamic_train_pit_sc/` | pit+sc model outputs |
| `outputs/dashboard/` | `dashboard_export.json` |
| `outputs/f1_dynamic_train/sensitivity/` | `summary.csv` |

---

## 🔭 Next Steps for Team (Phase III)

### 📊 1. Dashboard Refinement
- Add filters by season, weather, and track type
- Integrate `duality_metrics.json` live updates

### 🧠 2. Paper & Presentation
**Will:** Model architecture, pipeline figure, calibration analysis  
**Ashish:** Driver vs Constructor interpretation, case studies (e.g., Verstappen vs Leclerc)

**Include:**
- DCSI formula, Bayesian update diagram
- Time–Rank duality visuals
- Calibration curves and reliability plots

### ⚙️ 3. Future Enhancements
- Pull real FastF1 data once API cache is enabled
- Add track/weather-specific priors
- Integrate live odds updates via `odds_snapshots.csv`
- Publish interactive dashboard (Streamlit Cloud or Observable)

---

## 👥 Authors

**Will Arsenault & Ashish Sakhuja**  
Penn State Sports Analytics Project (2025)