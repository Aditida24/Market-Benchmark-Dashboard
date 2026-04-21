import io
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Market Benchmark & Tariff Strategy Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main > div {padding-top: 1rem;}
    .insight-box {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border-left: 5px solid #2563eb;
        background: rgba(37,99,235,0.08);
        margin-bottom: 0.8rem;
    }
    .small-note {color: #6b7280; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_SHEETS = {"All_Profile_Summary", "Profile_Overview", "DAM", "Settlement"}


# -----------------------------
# Helpers
# -----------------------------
def eur(x: float, digits: int = 0) -> str:
    if pd.isna(x):
        return "-"
    return f"€{x:,.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}%"


def num(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"


def safe_mean_abs_error(actual: pd.Series, predicted: pd.Series) -> float:
    df = pd.concat([actual, predicted], axis=1).dropna()
    if df.empty:
        return np.nan
    return np.mean(np.abs(df.iloc[:, 0] - df.iloc[:, 1]))


@st.cache_data(show_spinner=False)
def load_workbook(uploaded_file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(uploaded_file)
    missing = REQUIRED_SHEETS - set(xls.sheet_names)
    if missing:
        raise ValueError(f"Missing required sheets: {sorted(missing)}")

    data = {
        "summary": pd.read_excel(uploaded_file, sheet_name="All_Profile_Summary"),
        "profiles": pd.read_excel(uploaded_file, sheet_name="Profile_Overview"),
        "dam": pd.read_excel(uploaded_file, sheet_name="DAM"),
        "settlement": pd.read_excel(uploaded_file, sheet_name="Settlement"),
    }
    return prepare_data(data)


@st.cache_data(show_spinner=False)
def prepare_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    summary = data["summary"].copy()
    profiles = data["profiles"].copy()
    dam = data["dam"].copy()
    settlement = data["settlement"].copy()

    # Numeric cleanup
    for col in [
        "annual_total_cost_eur",
        "difference_vs_dam_10pct_eur",
        "difference_vs_settlement_10pct_eur",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    summary["rank"] = pd.to_numeric(summary.get("rank"), errors="coerce")
    summary["cheaper_than_market"] = summary.get("cheaper_than_market", False).fillna(False).astype(bool)

    for col in [
        "occupants", "floor_area_m2", "how_many_evs",
        "annual_appliances_kwh", "annual_heating_kwh", "annual_ev_kwh", "annual_total_kwh"
    ]:
        if col in profiles.columns:
            profiles[col] = pd.to_numeric(profiles[col], errors="coerce")

    # DAM sheet: original source is €/MWh, convert only this column to €/kWh for reporting
    dam["timestamp"] = pd.to_datetime(dam["start_time_utc"], errors="coerce", utc=True)
    dam["price_eur_mwh"] = pd.to_numeric(dam["price_eur_mwh"], errors="coerce")
    dam["price_eur_kwh"] = dam["price_eur_mwh"] / 1000
    dam["month"] = dam["timestamp"].dt.to_period("M").astype(str)
    dam["hour"] = dam["timestamp"].dt.hour
    dam["weekday"] = dam["timestamp"].dt.day_name()

    # Settlement sheet remains unchanged as requested
    settlement["timestamp"] = pd.to_datetime(settlement["timestamp"], errors="coerce", utc=True)
    settlement["settlement_price"] = pd.to_numeric(settlement["settlement_price"], errors="coerce")
    settlement["predicted_settlement_price"] = pd.to_numeric(settlement["predicted_settlement_price"], errors="coerce")
    settlement["month"] = settlement["timestamp"].dt.to_period("M").astype(str)
    settlement["hour"] = settlement["timestamp"].dt.hour
    settlement["weekday"] = settlement["timestamp"].dt.day_name()

    market = pd.merge(
        dam[["timestamp", "month", "hour", "weekday", "price_eur_kwh"]],
        settlement[["timestamp", "month", "hour", "weekday", "settlement_price", "predicted_settlement_price"]],
        on=["timestamp", "month", "hour", "weekday"],
        how="inner",
    )
    market["spread_settlement_minus_dam"] = market["settlement_price"] - market["price_eur_kwh"]
    market["spread_abs"] = market["spread_settlement_minus_dam"].abs()
    market["better_market"] = np.where(
        market["settlement_price"] < market["price_eur_kwh"],
        "Settlement",
        "DAM"
    )

    monthly_market = market.groupby("month", as_index=False).agg(
        avg_dam=("price_eur_kwh", "mean"),
        avg_settlement=("settlement_price", "mean"),
        avg_predicted_settlement=("predicted_settlement_price", "mean"),
        avg_spread=("spread_settlement_minus_dam", "mean"),
        avg_abs_spread=("spread_abs", "mean"),
    )

    hourly_market = market.groupby("hour", as_index=False).agg(
        avg_dam=("price_eur_kwh", "mean"),
        avg_settlement=("settlement_price", "mean"),
        avg_spread=("spread_settlement_minus_dam", "mean"),
    )

    weekday_market = market.groupby("weekday", as_index=False).agg(
        avg_dam=("price_eur_kwh", "mean"),
        avg_settlement=("settlement_price", "mean"),
        avg_spread=("spread_settlement_minus_dam", "mean"),
    )
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_market["weekday"] = pd.Categorical(weekday_market["weekday"], categories=weekday_order, ordered=True)
    weekday_market = weekday_market.sort_values("weekday")

    supplier_only = summary[summary["category"].eq("Supplier Tariff")].copy()
    benchmark_only = summary[summary["category"].eq("Market Reference")].copy()

    best_supplier = (
        supplier_only.sort_values(["profile_name", "annual_total_cost_eur"])
        .groupby("profile_name", as_index=False)
        .first()[[
            "profile_name", "provider_name", "tariff_option_name", "annual_total_cost_eur",
            "difference_vs_dam_10pct_eur", "difference_vs_settlement_10pct_eur", "cheaper_than_market"
        ]]
        .rename(columns={
            "provider_name": "best_supplier",
            "tariff_option_name": "best_supplier_tariff",
            "annual_total_cost_eur": "best_supplier_cost_eur",
            "difference_vs_dam_10pct_eur": "best_vs_dam_eur",
            "difference_vs_settlement_10pct_eur": "best_vs_settlement_eur",
            "cheaper_than_market": "best_cheaper_than_market",
        })
    )

    worst_supplier = (
        supplier_only.sort_values(["profile_name", "annual_total_cost_eur"], ascending=[True, False])
        .groupby("profile_name", as_index=False)
        .first()[["profile_name", "provider_name", "tariff_option_name", "annual_total_cost_eur"]]
        .rename(columns={
            "provider_name": "worst_supplier",
            "tariff_option_name": "worst_supplier_tariff",
            "annual_total_cost_eur": "worst_supplier_cost_eur",
        })
    )

    best_benchmark = (
        benchmark_only.sort_values(["profile_name", "annual_total_cost_eur"])
        .groupby("profile_name", as_index=False)
        .first()[["profile_name", "provider_name", "tariff_option_name", "annual_total_cost_eur"]]
        .rename(columns={
            "provider_name": "best_benchmark_provider",
            "tariff_option_name": "best_benchmark_option",
            "annual_total_cost_eur": "best_benchmark_cost_eur",
        })
    )

    profile_benchmark = (
        profiles.merge(best_supplier, on="profile_name", how="left")
        .merge(worst_supplier, on="profile_name", how="left")
        .merge(best_benchmark, on="profile_name", how="left")
    )
    profile_benchmark["switching_saving_eur"] = (
        profile_benchmark["worst_supplier_cost_eur"] - profile_benchmark["best_supplier_cost_eur"]
    )
    profile_benchmark["retail_premium_vs_benchmark_eur"] = (
        profile_benchmark["best_supplier_cost_eur"] - profile_benchmark["best_benchmark_cost_eur"]
    )
    profile_benchmark["best_option"] = np.where(
        profile_benchmark["best_supplier_cost_eur"] <= profile_benchmark["best_benchmark_cost_eur"],
        "Supplier Tariff",
        "Benchmark Reference",
    )

    provider_heatmap = (
        supplier_only.groupby(["profile_name", "provider_name"], as_index=False)["annual_total_cost_eur"]
        .min()
        .pivot(index="profile_name", columns="provider_name", values="annual_total_cost_eur")
    )

    benchmark_compare_long = profile_benchmark[[
        "profile_name", "best_supplier_cost_eur", "best_benchmark_cost_eur"
    ]].melt(
        id_vars="profile_name",
        var_name="type",
        value_name="annual_cost_eur"
    )
    benchmark_compare_long["type"] = benchmark_compare_long["type"].map({
        "best_supplier_cost_eur": "Best Supplier Tariff",
        "best_benchmark_cost_eur": "Best Benchmark",
    })

    return {
        "summary": summary,
        "profiles": profiles,
        "dam": dam,
        "settlement": settlement,
        "market": market,
        "monthly_market": monthly_market,
        "hourly_market": hourly_market,
        "weekday_market": weekday_market,
        "supplier_only": supplier_only,
        "benchmark_only": benchmark_only,
        "profile_benchmark": profile_benchmark,
        "provider_heatmap": provider_heatmap,
        "benchmark_compare_long": benchmark_compare_long,
    }


# -----------------------------
# Sidebar filters
# -----------------------------
def apply_filters(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    pb = data["profile_benchmark"].copy()

    st.sidebar.header("Filter Panel")

    profiles = sorted(pb["profile_name"].dropna().unique().tolist())
    building_types = sorted(pb["building_type"].dropna().unique().tolist()) if "building_type" in pb.columns else []
    heating_types = sorted(pb["heating_system"].dropna().unique().tolist()) if "heating_system" in pb.columns else []
    ev_counts = sorted(pb["how_many_evs"].dropna().astype(int).unique().tolist()) if "how_many_evs" in pb.columns else []
    occupant_counts = sorted(pb["occupants"].dropna().astype(int).unique().tolist()) if "occupants" in pb.columns else []

    selected_profiles = st.sidebar.multiselect("Household profile", profiles, default=profiles)
    selected_building = st.sidebar.multiselect("Building type", building_types, default=building_types)
    selected_heating = st.sidebar.multiselect("Heating system", heating_types, default=heating_types)
    selected_evs = st.sidebar.multiselect("EV count", ev_counts, default=ev_counts)
    selected_occupants = st.sidebar.multiselect("Occupants", occupant_counts, default=occupant_counts)

    months = sorted(data["monthly_market"]["month"].dropna().unique().tolist())
    selected_months = st.sidebar.multiselect("Market months", months, default=months)

    spread_focus = st.sidebar.selectbox(
        "Spread focus",
        ["All", "Settlement above DAM", "Settlement below DAM"],
        index=0,
    )

    filtered_pb = pb[
        pb["profile_name"].isin(selected_profiles)
        & pb["building_type"].isin(selected_building)
        & pb["heating_system"].isin(selected_heating)
        & pb["how_many_evs"].fillna(0).astype(int).isin(selected_evs)
        & pb["occupants"].fillna(0).astype(int).isin(selected_occupants)
    ].copy()

    market = data["market"].copy()
    market = market[market["month"].isin(selected_months)]
    if spread_focus == "Settlement above DAM":
        market = market[market["spread_settlement_minus_dam"] > 0]
    elif spread_focus == "Settlement below DAM":
        market = market[market["spread_settlement_minus_dam"] < 0]

    filtered_monthly = data["monthly_market"][data["monthly_market"]["month"].isin(selected_months)].copy()
    filtered_hourly = data["hourly_market"].copy()
    filtered_weekday = data["weekday_market"].copy()

    profile_names = filtered_pb["profile_name"].tolist()
    filtered_supplier_only = data["supplier_only"][data["supplier_only"]["profile_name"].isin(profile_names)].copy()
    filtered_benchmark_only = data["benchmark_only"][data["benchmark_only"]["profile_name"].isin(profile_names)].copy()
    filtered_benchmark_compare = data["benchmark_compare_long"][data["benchmark_compare_long"]["profile_name"].isin(profile_names)].copy()
    filtered_heatmap = data["provider_heatmap"].loc[data["provider_heatmap"].index.intersection(profile_names)] if profile_names else data["provider_heatmap"].iloc[0:0]

    return {
        "profile_benchmark": filtered_pb,
        "market": market,
        "monthly_market": filtered_monthly,
        "hourly_market": filtered_hourly,
        "weekday_market": filtered_weekday,
        "supplier_only": filtered_supplier_only,
        "benchmark_only": filtered_benchmark_only,
        "benchmark_compare_long": filtered_benchmark_compare,
        "provider_heatmap": filtered_heatmap,
        "spread_focus": spread_focus,
    }


# -----------------------------
# KPI metrics
# -----------------------------
def build_metrics(filtered: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    pb = filtered["profile_benchmark"]
    market = filtered["market"]
    supplier_only = filtered["supplier_only"]

    best_profile = pb.loc[pb["best_supplier_cost_eur"].idxmin()] if not pb.empty else None
    highest_saving_profile = pb.loc[pb["switching_saving_eur"].idxmax()] if not pb.empty else None
    highest_premium_profile = pb.loc[pb["retail_premium_vs_benchmark_eur"].idxmax()] if not pb.empty else None

    top_provider = (
        pb["best_supplier"].value_counts().idxmax() if not pb.empty and pb["best_supplier"].notna().any() else "-"
    )

    dam_better_pct = 100 * (market["better_market"].eq("DAM").mean()) if not market.empty else np.nan
    settlement_better_pct = 100 * (market["better_market"].eq("Settlement").mean()) if not market.empty else np.nan
    corr = market[["settlement_price", "price_eur_kwh"]].corr().iloc[0, 1] if len(market) > 1 else np.nan
    mae = safe_mean_abs_error(market["settlement_price"], market["predicted_settlement_price"]) if not market.empty else np.nan

    return {
        "avg_dam": market["price_eur_kwh"].mean() if not market.empty else np.nan,
        "avg_settlement": market["settlement_price"].mean() if not market.empty else np.nan,
        "avg_spread": market["spread_settlement_minus_dam"].mean() if not market.empty else np.nan,
        "avg_abs_spread": market["spread_abs"].mean() if not market.empty else np.nan,
        "corr_dam_settlement": corr,
        "pred_mae": mae,
        "dam_better_pct": dam_better_pct,
        "settlement_better_pct": settlement_better_pct,
        "top_provider": top_provider,
        "best_profile": best_profile,
        "highest_saving_profile": highest_saving_profile,
        "highest_premium_profile": highest_premium_profile,
        "tariff_count": supplier_only["tariff_option_name"].nunique() if not supplier_only.empty else 0,
    }


# -----------------------------
# Sections
# -----------------------------
def overview_section(filtered: Dict[str, pd.DataFrame], metrics: Dict[str, object]) -> None:
    st.subheader("1. DAM vs Settlement Price Comparison")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg DAM", eur(metrics["avg_dam"], 4) + "/kWh")
    c2.metric("Avg Settlement", eur(metrics["avg_settlement"], 4) + "/kWh")
    c3.metric("Avg Spread", eur(metrics["avg_spread"], 4) + "/kWh")
    c4.metric("Avg Absolute Spread", eur(metrics["avg_abs_spread"], 4) + "/kWh")
    c5.metric("Best Market More Often", "DAM" if (metrics["dam_better_pct"] or 0) >= (metrics["settlement_better_pct"] or 0) else "Settlement")

    monthly = filtered["monthly_market"]
    market = filtered["market"]

    left, right = st.columns([1.25, 1])
    with left:
        monthly_long = monthly.melt(
            id_vars="month",
            value_vars=["avg_dam", "avg_settlement", "avg_predicted_settlement"],
            var_name="series",
            value_name="price",
        )
        monthly_long["series"] = monthly_long["series"].map({
            "avg_dam": "DAM",
            "avg_settlement": "Settlement",
            "avg_predicted_settlement": "Predicted Settlement",
        })
        fig = px.line(
            monthly_long,
            x="month",
            y="price",
            color="series",
            markers=True,
            title="Monthly DAM vs Settlement Trend",
        )
        fig.update_layout(height=430, xaxis_title="Month", yaxis_title="€/kWh")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        hourly = filtered["hourly_market"]
        fig_hour = px.bar(
            hourly,
            x="hour",
            y="avg_spread",
            title="Average Settlement - DAM Spread by Hour",
            labels={"hour": "Hour of Day", "avg_spread": "Spread (€/kWh)"},
        )
        fig_hour.update_layout(height=430)
        st.plotly_chart(fig_hour, use_container_width=True)

    if not market.empty:
        compare_choice = pd.DataFrame({
            "Market Option": ["DAM cheaper", "Settlement cheaper"],
            "Share %": [metrics["dam_better_pct"], metrics["settlement_better_pct"]],
        })
        fig_choice = px.bar(
            compare_choice,
            x="Market Option",
            y="Share %",
            title="Which Market Is Lower More Often?",
            text="Share %",
        )
        fig_choice.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_choice.update_layout(height=360)
        st.plotly_chart(fig_choice, use_container_width=True)

    recommendation = "DAM" if (metrics["dam_better_pct"] or 0) > (metrics["settlement_better_pct"] or 0) else "Settlement"
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Business readout**

- On the selected market window, **{recommendation}** is the lower-priced option more often.
- Average DAM price is **{eur(metrics['avg_dam'], 4)}/kWh**, while average Settlement is **{eur(metrics['avg_settlement'], 4)}/kWh**.
- The average absolute spread is **{eur(metrics['avg_abs_spread'], 4)}/kWh**, which shows how far balancing outcomes move away from the day-ahead signal.
- This helps answer: **Should we rely more on DAM or Settlement as the benchmark when judging tariffs?**
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)



def tariff_section(filtered: Dict[str, pd.DataFrame], metrics: Dict[str, object]) -> None:
    st.subheader("2. Best Tariff Options to Choose")
    pb = filtered["profile_benchmark"]
    supplier_only = filtered["supplier_only"]

    if pb.empty:
        st.warning("No household profiles match the current filters.")
        return

    selected_profile = st.selectbox("Choose a household profile", pb["profile_name"].tolist())
    profile_table = supplier_only[supplier_only["profile_name"].eq(selected_profile)].sort_values("annual_total_cost_eur")
    profile_summary = pb[pb["profile_name"].eq(selected_profile)].iloc[0]

    left, right = st.columns([1.2, 1])
    with left:
        fig_provider = px.bar(
            profile_table,
            x="provider_name",
            y="annual_total_cost_eur",
            color="provider_name",
            hover_data=["tariff_option_name", "difference_vs_dam_10pct_eur", "difference_vs_settlement_10pct_eur"],
            title="Provider Cost Ranking for Selected Profile",
            labels={"annual_total_cost_eur": "Annual Cost (€)", "provider_name": "Provider"},
        )
        fig_provider.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig_provider, use_container_width=True)

    with right:
        comp = pd.DataFrame({
            "Measure": ["Best Supplier", "Worst Supplier", "Benchmark"],
            "Annual Cost (€)": [
                profile_summary["best_supplier_cost_eur"],
                profile_summary["worst_supplier_cost_eur"],
                profile_summary["best_benchmark_cost_eur"],
            ]
        })
        fig_comp = px.bar(
            comp,
            x="Measure",
            y="Annual Cost (€)",
            color="Measure",
            title="Best Supplier vs Worst Supplier vs Benchmark",
        )
        fig_comp.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Tariff recommendation for {selected_profile}**

- Recommended supplier: **{profile_summary['best_supplier']}**
- Recommended tariff option: **{profile_summary['best_supplier_tariff']}**
- Best supplier cost: **{eur(profile_summary['best_supplier_cost_eur'])}**
- Worst supplier cost: **{eur(profile_summary['worst_supplier_cost_eur'])}**
- Switching opportunity: **{eur(profile_summary['switching_saving_eur'])}**
- Versus DAM benchmark: **{eur(profile_summary['best_vs_dam_eur'])}**
- Versus Settlement benchmark: **{eur(profile_summary['best_vs_settlement_eur'])}**
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    winner_table = pb[[
        "profile_name", "building_type", "heating_system", "how_many_evs", "annual_total_kwh",
        "best_supplier", "best_supplier_tariff", "best_supplier_cost_eur",
        "switching_saving_eur", "best_option"
    ]].copy().sort_values("best_supplier_cost_eur")
    winner_table = winner_table.rename(columns={
        "profile_name": "Profile",
        "building_type": "Building Type",
        "heating_system": "Heating",
        "how_many_evs": "EVs",
        "annual_total_kwh": "Annual kWh",
        "best_supplier": "Best Supplier",
        "best_supplier_tariff": "Best Tariff",
        "best_supplier_cost_eur": "Best Supplier Cost (€)",
        "switching_saving_eur": "Switching Saving (€)",
        "best_option": "Cheapest Overall Option",
    })
    st.dataframe(winner_table, use_container_width=True, hide_index=True)



def benchmark_section(filtered: Dict[str, pd.DataFrame], metrics: Dict[str, object]) -> None:
    st.subheader("3. Benchmark Comparisons")
    pb = filtered["profile_benchmark"]
    if pb.empty:
        st.warning("No benchmark rows available for the selected filters.")
        return

    left, right = st.columns([1.15, 1])
    with left:
        fig_bench = px.bar(
            filtered["benchmark_compare_long"],
            x="profile_name",
            y="annual_cost_eur",
            color="type",
            barmode="group",
            title="Best Supplier Tariff vs Best Benchmark by Profile",
            labels={"profile_name": "Profile", "annual_cost_eur": "Annual Cost (€)", "type": "Option"},
        )
        fig_bench.update_layout(height=430)
        st.plotly_chart(fig_bench, use_container_width=True)

    with right:
        fig_gap = px.bar(
            pb.sort_values("retail_premium_vs_benchmark_eur", ascending=True),
            x="retail_premium_vs_benchmark_eur",
            y="profile_name",
            orientation="h",
            color="best_option",
            title="Retail Premium vs Best Benchmark",
            labels={"retail_premium_vs_benchmark_eur": "Premium (€)", "profile_name": "Profile", "best_option": "Cheapest Option"},
        )
        fig_gap.update_layout(height=430)
        st.plotly_chart(fig_gap, use_container_width=True)

    heatmap = filtered["provider_heatmap"]
    if not heatmap.empty:
        fig_heat = px.imshow(
            heatmap,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale="RdYlGn_r",
            title="Supplier Benchmark Heatmap by Profile",
        )
        fig_heat.update_layout(height=420, coloraxis_colorbar_title="€/year")
        st.plotly_chart(fig_heat, use_container_width=True)

    top_gap = pb.loc[pb["retail_premium_vs_benchmark_eur"].idxmax()]
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Benchmark insight**

- The profile with the highest gap versus benchmark is **{top_gap['profile_name']}**.
- Its best supplier cost is **{eur(top_gap['best_supplier_cost_eur'])}**, versus a best benchmark of **{eur(top_gap['best_benchmark_cost_eur'])}**.
- This premium of **{eur(top_gap['retail_premium_vs_benchmark_eur'])}** highlights where the strongest pricing pressure or negotiation opportunity sits.
- This section helps answer: **Are supplier tariffs competitive enough relative to market-based references?**
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)



def solution_section(filtered: Dict[str, pd.DataFrame], metrics: Dict[str, object]) -> None:
    st.subheader("4. Business Problems, Insights, and Recommended Actions")
    pb = filtered["profile_benchmark"]
    market = filtered["market"]

    if pb.empty:
        st.warning("No profiles available for recommendation output.")
        return

    highest_saving = pb.loc[pb["switching_saving_eur"].idxmax()]
    highest_usage = pb.loc[pb["annual_total_kwh"].idxmax()]
    most_evs = pb.sort_values("how_many_evs", ascending=False).iloc[0]

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Relevant business problems this dashboard can answer**

1. **Which market reference should we trust more for pricing comparison?**  
   Use the DAM vs Settlement section. In the current view, DAM is cheaper **{pct(metrics['dam_better_pct'])}** of the time and Settlement is cheaper **{pct(metrics['settlement_better_pct'])}** of the time.

2. **Which household profile has the highest switching opportunity?**  
   **{highest_saving['profile_name']}** with a potential saving of **{eur(highest_saving['switching_saving_eur'])}**.

3. **Which households should be prioritized for tariff optimization?**  
   High-usage households like **{highest_usage['profile_name']}** at **{num(highest_usage['annual_total_kwh'], 0)} kWh/year**.

4. **Where do EV households create a case for smarter pricing?**  
   Profiles like **{most_evs['profile_name']}** can be used to test time-based pricing strategies and off-peak charging recommendations.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    recommendations = pb[[
        "profile_name", "building_type", "heating_system", "how_many_evs", "occupants",
        "annual_total_kwh", "best_supplier", "best_supplier_tariff", "best_supplier_cost_eur",
        "best_benchmark_option", "best_benchmark_cost_eur", "retail_premium_vs_benchmark_eur",
        "switching_saving_eur", "best_option"
    ]].copy().sort_values(["switching_saving_eur", "annual_total_kwh"], ascending=[False, False])

    recommendations = recommendations.rename(columns={
        "profile_name": "Profile",
        "building_type": "Building Type",
        "heating_system": "Heating",
        "how_many_evs": "EVs",
        "occupants": "Occupants",
        "annual_total_kwh": "Annual kWh",
        "best_supplier": "Best Supplier",
        "best_supplier_tariff": "Best Tariff",
        "best_supplier_cost_eur": "Best Supplier Cost (€)",
        "best_benchmark_option": "Best Benchmark Option",
        "best_benchmark_cost_eur": "Best Benchmark Cost (€)",
        "retail_premium_vs_benchmark_eur": "Premium vs Benchmark (€)",
        "switching_saving_eur": "Switching Saving (€)",
        "best_option": "Cheapest Overall Option",
    })

    st.dataframe(recommendations, use_container_width=True, hide_index=True)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        recommendations.to_excel(writer, sheet_name="strategy_output", index=False)
    output.seek(0)

    st.download_button(
        "Download strategy table",
        data=output,
        file_name="market_benchmark_strategy_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------
# Main app
# -----------------------------
st.title("📈 Market Benchmark & Tariff Strategy Dashboard")
st.caption("Business-focused dashboard for DAM vs Settlement comparison, tariff selection, and benchmark strategy.")

DEFAULT_FILE = "Dataset _Dashboard.xlsx"

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a different workbook", type=["xlsx"])

file_source = uploaded_file if uploaded_file is not None else DEFAULT_FILE

try:
    data = load_workbook(file_source)
except Exception as e:
    st.error(f"Could not load workbook: {e}")
    st.stop()

filtered = apply_filters(data)
metrics = build_metrics(filtered)

section1, section2, section3, section4 = st.tabs([
    "Market Comparison",
    "Tariff Selection",
    "Benchmark Comparison",
    "Business Solutions",
])

with section1:
    overview_section(filtered, metrics)

with section2:
    tariff_section(filtered, metrics)

with section3:
    benchmark_section(filtered, metrics)

with section4:
    solution_section(filtered, metrics)

st.markdown("---")
st.markdown(
    '<div class="small-note">This app is designed for workbooks with the same structure and gives meeting-ready filters, comparisons, and business insights.</div>',
    unsafe_allow_html=True,
)
