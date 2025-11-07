import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="F1 Driver–Constructor Dashboard",
    page_icon="🏎️",
    layout="wide"
)

DASHBOARD_PATH = Path("outputs/dashboard/dashboard_export.json")

# ----------------------------
# Load JSON
# ----------------------------
if not DASHBOARD_PATH.exists():
    st.error(f"❌ Dashboard export not found at {DASHBOARD_PATH}")
    st.stop()

with open(DASHBOARD_PATH, "r") as f:
    dashboard = json.load(f)

meta = dashboard.get("meta", {})
drivers = dashboard.get("top_outperforming_drivers", [])
teams = dashboard.get("team_strength_rankings", [])
driver_ts = dashboard.get("driver_dcsi_timeseries", [])
team_ts = dashboard.get("team_dcsi_timeseries", [])
calibration = dashboard.get("calibration_metrics", {})
duality = dashboard.get("duality_metrics", {})

# Convert to DataFrames for visualization
drv_df = pd.DataFrame(drivers)
tm_df = pd.DataFrame(teams)
drv_ts = pd.DataFrame(driver_ts)
tm_ts = pd.DataFrame(team_ts)

# ----------------------------
# Layout
# ----------------------------
st.title("🏎️ Formula 1 Driver–Constructor Dashboard")
st.caption("Phase II • Dynamic Bayesian DCSI Analysis (2025)")

st.markdown(f"**Source Folder:** `{meta.get('in_dir', '')}`")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏁 Top Drivers",
    "🏭 Top Constructors",
    "📈 DCSI Time Series",
    "🎯 Calibration & Duality"
])

# ----------------------------
# Tab 1 – Drivers
# ----------------------------
with tab1:
    st.subheader("Top 10 'Outperforming Car' Drivers")
    if not drv_df.empty:
        fig = px.bar(
            drv_df.sort_values("outperforming_car", ascending=True),
            x="outperforming_car",
            y="driver_name",
            orientation="h",
            color="outperforming_car",
            color_continuous_scale="bluered_r",
            title="Driver Effect (Higher = Outperforming Car)"
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(drv_df)
    else:
        st.info("No driver data found. Run `export_dashboard_json.py` first.")

# ----------------------------
# Tab 2 – Constructors
# ----------------------------
with tab2:
    st.subheader("Constructor Strength Ranking")
    if not tm_df.empty:
        fig = px.bar(
            tm_df.sort_values("team_strength", ascending=True),
            x="team_strength",
            y="constructor_name",
            orientation="h",
            color="team_strength",
            color_continuous_scale="sunset",
            title="Constructor Effect (Higher = Stronger Car)"
        )
        fig.update_layout(yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tm_df)
    else:
        st.info("No constructor data found.")

# ----------------------------
# Tab 3 – Time Series (Driver & Team)
# ----------------------------
with tab3:
    st.subheader("Driver DCSI Over Time")
    if not drv_ts.empty:
        selected_driver = st.selectbox("Select Driver:", sorted(drv_ts["driver_name"].unique()))
        subset = drv_ts[drv_ts["driver_name"] == selected_driver].copy()

        # choose correct x-axis automatically
        x_col = "round" if "round" in subset.columns else "season_id"
        subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
        subset = subset.sort_values(x_col)

        fig = px.line(
            subset,
            x=x_col,
            y="driver_eff_mean",
            color="season_id",
            markers=True,
            title=f"DCSI Trend – {selected_driver}"
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(
            xaxis_title=x_col.capitalize(),
            yaxis_title="Driver Effect (DCSI)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No driver time series data found.")

    st.subheader("Team DCSI Over Time")
    if not tm_ts.empty:
        selected_team = st.selectbox("Select Constructor:", sorted(tm_ts["constructor_name"].unique()))
        subset_t = tm_ts[tm_ts["constructor_name"] == selected_team].copy()

        # choose correct x-axis automatically
        x_col_t = "round" if "round" in subset_t.columns else "season_id"
        subset_t[x_col_t] = pd.to_numeric(subset_t[x_col_t], errors="coerce")
        subset_t = subset_t.sort_values(x_col_t)

        fig_t = px.line(
            subset_t,
            x=x_col_t,
            y="team_eff_mean",
            color_discrete_sequence=["#1f77b4"],
            markers=True,
            title=f"Constructor Strength Trend – {selected_team}"
        )
        fig_t.update_traces(mode="lines+markers")
        fig_t.update_layout(
            xaxis_title=x_col_t.capitalize(),
            yaxis_title="Team Effect (DCSI)",
            showlegend=False
        )
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No constructor time series data found.")

# ----------------------------
# Tab 4 – Calibration & Duality
# ----------------------------
with tab4:
    st.subheader("Calibration Metrics (Brier & Log Loss)")
    flat_cal = []

    if isinstance(calibration, dict):
        for cat, vals in calibration.items():
            if isinstance(vals, dict):
                for met, val in vals.items():
                    if isinstance(val, (int, float)):
                        flat_cal.append({"Metric": f"{cat}-{met}", "Value": val})
            elif isinstance(vals, (int, float)):
                flat_cal.append({"Metric": cat, "Value": vals})
    elif isinstance(calibration, list):
        flat_cal = calibration
    elif isinstance(calibration, str):
        st.info(f"Calibration metrics path: {calibration}")
    else:
        st.info("No calibration metrics found in JSON.")

    if flat_cal:
        cal_df = pd.DataFrame(flat_cal)
        fig = px.bar(
            cal_df,
            x="Value",
            y="Metric",
            orientation="h",
            color="Value",
            color_continuous_scale="RdYlGn_r",
            title="Calibration Overview"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cal_df)
    else:
        st.info("No numeric calibration values found.")

    st.subheader("Time–Rank Duality Metrics")
    if isinstance(duality, dict):
        dual_df = pd.DataFrame(duality.get("metrics", []))
        if not dual_df.empty:
            st.dataframe(dual_df)
            if "win_softmax" in dual_df.columns and "win_pl" in dual_df.columns:
                fig = px.scatter(
                    dual_df,
                    x="win_softmax",
                    y="win_pl",
                    trendline="ols",
                    title="Softmax vs Plackett–Luce Agreement"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No duality metrics found.")
    else:
        st.info("Duality data missing or invalid.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Built by **Will Arsenault & Ashish Sakhuja** — Penn State Sports Analytics Project")
