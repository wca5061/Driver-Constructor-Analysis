# F1 Driver–Constructor Separation (DCSI) – README

This README shows you **exactly** how to run the project end-to-end in the right order with the correct files, environment variables, and expected outputs. Follow it line-by-line.

---

## 0) Setup

### Requirements


```bash
# from repo root
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt      # if present
# otherwise:
pip install numpy pandas matplotlib arviz pymc scipy fastf1
Paths (defaults)
Data: data/synth_f1_2018_2025_realish/
```

Splits: `outputs/splits/splits.csv`

Dynamic (TRAIN): `outputs/f1_dynamic_train/`

Dynamic (TEST): `outputs/f1_dynamic_test/`

Parent model: `outputs/f1_parent_model/`

You don’t need to create these folders manually; the scripts will.

1) Build the synthetic dataset

```
python dataset_builder.py
```

You should see: CSVs in data/synth_f1_2018_2025_realish/ including:
races.csv, race_entries.csv, drivers.csv, constructors.csv, laps.csv, pitstops.csv, safetycars.csv, odds_snapshots.csv.

2) Create the 70/20/10 splits (TRAIN/TEST/VAL)
bash
```
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs" \
python create_splits.py
```
Produces: outputs/splits/splits.csv

3) Run the dynamic model on TRAIN (fits & saves priors)
bash
```
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python f1_dynamic_update.py
```
Produces (TRAIN folder):

dcsi_race.csv — per-race predictions & effects

dcsi_cumulative_drivers.csv, dcsi_cumulative_constructors.csv

priors.npz

figures in figs/

Quick sanity check:

bash
```
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python quick_check_dynamic.py
```

4) Predict on TEST (no fitting; uses TRAIN priors)
bash
```
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
F1_PRIORS_PATH="outputs/f1_dynamic_train/priors.npz" \
python f1_dynamic_update.py
```
Produces (TEST folder):
```
dcsi_race.csv
```
5) Convert scores → probabilities (TEST) with frozen τ (from TRAIN)

First, compute τ on TRAIN:

bash
```
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python make_probs_from_dynamic.py
Then use that same τ on TEST:
```
bash
```
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
F1_TAU_PATH="outputs/f1_dynamic_train/prob_metrics.json" \
python make_probs_from_dynamic.py
```
Produces (TEST):

dcsi_probs.csv (p(win), p(podium), p(points))

prob_metrics.json

calibration figures in figs/

6) Build explainable roll-ups (TEST)
bash
```
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
python build_rollups_explainable.py
```
Produces:

rollup_drivers_overall.csv, rollup_drivers_by_season.csv, rollup_drivers_condition_split.csv

rollup_teams_overall.csv, rollup_teams_by_season.csv

plots in figs/

Optional quick readout:

bash
```
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
python quick_check_dynamic.py
```
7) Time–Rank Duality evaluation (TEST) with frozen τ
bash
```
F1_OUT_DIR="outputs/f1_dynamic_test" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
F1_TAU_PATH="outputs/f1_dynamic_train/prob_metrics.json" \
python time_rank_duality.py
```
Produces:

duality_probs.csv (time/softmax vs PL probabilities side-by-side)

duality_metrics.json

calibration & method-agreement plots in figs/

8) (Optional) Parent model path
TRAIN fit:

bash
```
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_parent_model" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TRAIN \
python f1_parent_model.py
```
TEST predict using TRAIN posterior:

bash
```
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_parent_model" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
F1_SPLIT_TARGET=TEST \
python f1_parent_model.py
```
Diagnostics (works for either pipeline):

bash
```
F1_OUT_DIR="outputs/f1_dynamic_test"  F1_SPLITS_CSV="outputs/splits/splits.csv" F1_SPLIT_TARGET=TEST  python f1_diagnostics.py
# or
F1_OUT_DIR="outputs/f1_parent_model" F1_SPLITS_CSV="outputs/splits/splits.csv" F1_SPLIT_TARGET=TRAIN python f1_diagnostics.py
```
9) (Optional) Sensitivity sweep (TRAIN only)
bash
```
F1_DATA_DIR="data/synth_f1_2018_2025_realish" \
F1_OUT_DIR="outputs/f1_dynamic_train" \
F1_SPLITS_CSV="outputs/splits/splits.csv" \
python sensitivity_sweep.py
```
Produces: outputs/f1_dynamic_train/sensitivity/ (baseline + variants + summary.csv with rank-stability metrics)

Expected Outputs (Checklist)
outputs/splits/splits.csv

outputs/f1_dynamic_train/ → dcsi_race.csv, priors.npz, prob_metrics.json, figs/

outputs/f1_dynamic_test/ → dcsi_race.csv, dcsi_probs.csv, roll-ups CSVs, figs/

(Optional) outputs/f1_parent_model/ → posterior.nc, posterior_*.csv, dcsi_race.csv, figs/

## Next Steps (Phase II)

Now that the full dynamic + split-aware pipeline is working, here’s what we need to do next.

### 1. Verify Outputs & Sanity Check
- Open `outputs/f1_dynamic_train/dcsi_cumulative_drivers.csv` and confirm top drivers look realistic (high-performing constructors but not perfect correlation).
- Open `outputs/f1_dynamic_test/dcsi_probs.csv` — check that win/podium/points probabilities roughly sum to realistic values (no driver >0.8 win prob in a balanced race).
- Use `quick_check_dynamic.py` and `build_rollups_explainable.py` to visually confirm ranking and calibration are sensible.

---

### 2. Add Pit Stop & Safety Car Effects
Goal: quantify how much pit stop execution and safety cars influence results.

- Modify `f1_dynamic_update.py` to include:
  ```python
  F1_USE_PIT=1
  F1_USE_SC=1
Add regressors:

avg_pit_time_diff (per race, per team)

sc_laps (safety car laps normalized by total)

Rerun f1_dynamic_update.py for TRAIN and TEST.

Compare DCSI changes and re-run make_probs_from_dynamic.py for recalibration.

Output Goal: Understand which drivers gain or lose performance under chaotic race conditions.

3. Cross-Split Generalization & Robustness
Compare calibration metrics between TRAIN and TEST (prob_metrics.json files).

Plot Brier and log-loss differences.

Identify drivers or teams whose estimated effects drop significantly when tested out-of-sample.

4. Export Dashboard-Ready JSONs
Create a lightweight export for visualization:

bash
```
python export_dashboard_json.py
```
That script (you can add next) should collect:

Top 10 “Outperforming Car” drivers

Team strength rankings

Driver/constructor DCSI time series

Calibration and duality metrics

This can power a Streamlit or Dash front-end.

5. Time–Rank Duality Analysis (already working)
Use time_rank_duality.py (with frozen τ) to validate that time-based hazard probabilities ≈ rank-based PL probabilities.

Examine duality_metrics.json for consistency between methods.

Goal: Confirm that time-based and rank-based models agree within a few percentage points — supports our theoretical foundation.

6. Sensitivity & Stability
Run:

bash
```
python sensitivity_sweep.py
```
to test robustness of DCSI to noise and priors.

Review outputs/f1_dynamic_train/sensitivity/summary.csv.

Goal: Prove model stability under small perturbations in hyperparameters.

7. Paper & Presentation Prep
Split writing tasks:

Will: Model architecture, pipeline figure, calibration analysis.

Ashish: Driver vs Constructor interpretation, case studies (e.g., Verstappen outperforming vs Leclerc carried).

Include:

Formula for DCSI decomposition

Bayesian update diagram

Time–Rank duality figure

Calibration and reliability curves

Target figures: trace_core.png, drivers_outperforming_cars_bar.png, prob_calibration_win.png, and duality_scatter_win_softmax_vs_pl.png.

8. Future Enhancements
Integrate real FastF1 data (once the API cache path works)

Add weather- and track-type–specific hyperpriors

Introduce live betting model updates using odds_snapshots.csv

Publish results and visualizations via Streamlit or Observable dashboard

