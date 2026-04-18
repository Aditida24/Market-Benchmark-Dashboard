import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Tariff Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.2rem;
    }
    .kpi-card {
        padding: 0.85rem 1rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 18px;
        background: rgba(255,255,255,0.02);
    }
    .insight-box {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border-left: 4px solid #4F8BF9;
        background: rgba(79,139,249,0.08);
        margin-bottom: 0.8rem;
    }
    .small-note {
        color: #6b7280;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

REQUIRED_SHEETS = {
    "All_Profile_Summary",
    "Profile_Overview",
    "DAM",
    "Settlement",
}

SUMMARY_REQUIRED_COLS = {
    "profile_name",
    "rank",
    "category",
    "provider_name",
    "tariff_option_name",
    "annual_total_cost_eur",
    "difference_vs_dam_10pct_eur",
    "difference_vs_settlement_10pct_eur",
    "cheaper_than_market",
}

PROFILE_REQUIRED_COLS = {
    "profile_name",
    "building_type",
    "occupants",
    "heating_system",
    "floor_area_m2",
    "how_many_evs",
    "annual_appliances_kwh",
    "annual_heating_kwh",
    "annual_ev_kwh",
    "annual_total_kwh",
}

DAM_REQUIRED_COLS = {
    "trading_day",
    "period",
    "start_time_utc",
    "price_eur_mwh",
}

SETTLEMENT_REQUIRED_COLS = {
    "timestamp",
    "settlement_price",
    "predicted_settlement_price",
}


# -----------------------------
# Helpers
# -----------------------------
def eur(x: float, digits: int = 0) -> str:
    if pd.isna(x):
        return "-"
    return f"€{x:,.{digits}f}"


def num(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.{digits}f}%"


def safe_div(a: float, b: float) -> float:
    if b in [0, None] or pd.isna(b):
        return np.nan
    return a / b


@st.cache_data(show_spinner=False)
def load_workbook(uploaded_file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(uploaded_file)
    missing_sheets = REQUIRED_SHEETS - set(xls.sheet_names)
    if missing_sheets:
        raise ValueError(f"Missing required sheets: {sorted(missing_sheets)}")

    sheets = {
        "summary": pd.read_excel(uploaded_file, sheet_name="All_Profile_Summary"),
        "profiles": pd.read_excel(uploaded_file, sheet_name="Profile_Overview"),
        "dam": pd.read_excel(uploaded_file, sheet_name="DAM"),
        "settlement": pd.read_excel(uploaded_file, sheet_name="Settlement"),
    }
    validate_structure(sheets)
    return prepare_data(sheets)



def validate_structure(sheets: Dict[str, pd.DataFrame]) -> None:
    summary_cols = set(sheets["summary"].columns)
    profile_cols = set(sheets["profiles"].columns)
    dam_cols = set(sheets["dam"].columns)
    settlement_cols = set(sheets["settlement"].columns)

    missing = {
        "All_Profile_Summary": sorted(SUMMARY_REQUIRED_COLS - summary_cols),
        "Profile_Overview": sorted(PROFILE_REQUIRED_COLS - profile_cols),
        "DAM": sorted(DAM_REQUIRED_COLS - dam_cols),
        "Settlement": sorted(SETTLEMENT_REQUIRED_COLS - settlement_cols),
    }
    problems = {k: v for k, v in missing.items() if v}
    if problems:
        raise ValueError(f"Workbook structure issue: {problems}")



def prepare_data(sheets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    summary = sheets["summary"].copy()
    profiles = sheets["profiles"].copy()
    dam = sheets["dam"].copy()
    settlement = sheets["settlement"].copy()

    # Summary cleaning
    summary["annual_total_cost_eur"] = pd.to_numeric(summary["annual_total_cost_eur"], errors="coerce")
    summary["difference_vs_dam_10pct_eur"] = pd.to_numeric(summary["difference_vs_dam_10pct_eur"], errors="coerce")
    summary["difference_vs_settlement_10pct_eur"] = pd.to_numeric(summary["difference_vs_settlement_10pct_eur"], errors="coerce")
    summary["cheaper_than_market"] = summary["cheaper_than_market"].fillna(False).astype(bool)

    # Profile cleaning
    numeric_profile_cols = [
        "occupants", "floor_area_m2", "how_many_evs", "annual_appliances_kwh",
        "annual_heating_kwh", "annual_ev_kwh", "annual_total_kwh"
    ]
    for col in numeric_profile_cols:
        profiles[col] = pd.to_numeric(profiles[col], errors="coerce")

    profiles["appliances_share_pct"] = 100 * profiles["annual_appliances_kwh"] / profiles["annual_total_kwh"]
    profiles["heating_share_pct"] = 100 * profiles["annual_heating_kwh"] / profiles["annual_total_kwh"]
    profiles["ev_share_pct"] = 100 * profiles["annual_ev_kwh"] / profiles["annual_total_kwh"]
    profiles["dominant_load"] = profiles[["annual_appliances_kwh", "annual_heating_kwh", "annual_ev_kwh"]].idxmax(axis=1)
    profiles["dominant_load"] = profiles["dominant_load"].map({
        "annual_appliances_kwh": "Appliances",
        "annual_heating_kwh": "Heating",
        "annual_ev_kwh": "EV Charging",
    })

    # DAM cleaning
    dam["timestamp"] = pd.to_datetime(dam["start_time_utc"], errors="coerce", utc=True)
    dam["price_eur_mwh"] = pd.to_numeric(dam["price_eur_mwh"], errors="coerce")
    dam["month"] = dam["timestamp"].dt.to_period("M").astype(str)

    # Settlement cleaning
    settlement["timestamp"] = pd.to_datetime(settlement["timestamp"], errors="coerce", utc=True)
    settlement["settlement_price"] = pd.to_numeric(settlement["settlement_price"], errors="coerce")
    settlement["predicted_settlement_price"] = pd.to_numeric(settlement["predicted_settlement_price"], errors="coerce")
    settlement["month"] = settlement["timestamp"].dt.to_period("M").astype(str)

    supplier_summary = summary[summary["category"].eq("Supplier Tariff")].copy()
    market_summary = summary[summary["category"].eq("Market Reference")].copy()

    # Profile/provider view
    provider_profile_cost = (
        supplier_summary
        .groupby(["profile_name", "provider_name"], as_index=False)["annual_total_cost_eur"]
        .min()
    )

    # Best supplier per profile
    idx_best_supplier = supplier_summary.groupby("profile_name")["annual_total_cost_eur"].idxmin()
    best_supplier_per_profile = (
        supplier_summary.loc[idx_best_supplier, [
            "profile_name", "provider_name", "tariff_option_name", "annual_total_cost_eur",
            "difference_vs_dam_10pct_eur", "difference_vs_settlement_10pct_eur", "cheaper_than_market"
        ]]
        .rename(columns={
            "provider_name": "best_supplier",
            "tariff_option_name": "best_tariff",
            "annual_total_cost_eur": "best_supplier_cost_eur",
            "difference_vs_dam_10pct_eur": "best_diff_vs_dam_eur",
            "difference_vs_settlement_10pct_eur": "best_diff_vs_settlement_eur",
            "cheaper_than_market": "best_cheaper_than_market",
        })
        .reset_index(drop=True)
    )

    # Best market reference per profile
    idx_best_market = market_summary.groupby("profile_name")["annual_total_cost_eur"].idxmin()
    best_market_per_profile = (
        market_summary.loc[idx_best_market, ["profile_name", "provider_name", "tariff_option_name", "annual_total_cost_eur"]]
        .rename(columns={
            "provider_name": "best_market_reference",
            "tariff_option_name": "best_market_option",
            "annual_total_cost_eur": "best_market_cost_eur",
        })
        .reset_index(drop=True)
    )

    # Tariff spread
    tariff_spread = (
        supplier_summary.groupby("profile_name", as_index=False)
        .agg(
            cheapest_supplier_cost_eur=("annual_total_cost_eur", "min"),
            avg_supplier_cost_eur=("annual_total_cost_eur", "mean"),
            most_expensive_supplier_cost_eur=("annual_total_cost_eur", "max"),
            number_of_supplier_options=("tariff_option_name", "count"),
        )
    )
    tariff_spread["switching_saving_eur"] = (
        tariff_spread["most_expensive_supplier_cost_eur"] - tariff_spread["cheapest_supplier_cost_eur"]
    )

    profile_insights = (
        profiles.merge(best_supplier_per_profile, on="profile_name", how="left")
        .merge(best_market_per_profile, on="profile_name", how="left")
        .merge(tariff_spread, on="profile_name", how="left")
    )
    profile_insights["retail_premium_vs_market_eur"] = (
        profile_insights["best_supplier_cost_eur"] - profile_insights["best_market_cost_eur"]
    )
    profile_insights["switching_saving_pct_of_cheapest"] = 100 * (
        profile_insights["switching_saving_eur"] / profile_insights["cheapest_supplier_cost_eur"]
    )

    # Monthly market table
    monthly_dam = dam.groupby("month", as_index=False)["price_eur_mwh"].mean().rename(columns={"price_eur_mwh": "avg_dam_price_eur_mwh"})
    monthly_settlement = settlement.groupby("month", as_index=False).agg(
        avg_settlement_price_eur_mwh=("settlement_price", "mean"),
        avg_predicted_settlement_price_eur_mwh=("predicted_settlement_price", "mean"),
    )
    monthly_market = monthly_dam.merge(monthly_settlement, on="month", how="outer").sort_values("month")

    # Long market trend sample for faster plotting
    dam_plot = dam[["timestamp", "price_eur_mwh"]].dropna().rename(columns={"price_eur_mwh": "price"})
    dam_plot["series"] = "DAM"
    settlement_plot = settlement[["timestamp", "settlement_price"]].dropna().rename(columns={"settlement_price": "price"})
    settlement_plot["series"] = "Settlement"
    market_plot = pd.concat([dam_plot, settlement_plot], ignore_index=True).sort_values("timestamp")

    # Heatmap data
    heatmap_df = provider_profile_cost.pivot(index="profile_name", columns="provider_name", values="annual_total_cost_eur")

    return {
        "summary": summary,
        "supplier_summary": supplier_summary,
        "market_summary": market_summary,
        "profiles": profiles,
        "profile_insights": profile_insights,
        "provider_profile_cost": provider_profile_cost,
        "tariff_spread": tariff_spread,
        "dam": dam,
        "settlement": settlement,
        "monthly_market": monthly_market,
        "market_plot": market_plot,
        "heatmap_df": heatmap_df,
    }



def build_overview_metrics(profile_insights: pd.DataFrame, dam: pd.DataFrame, settlement: pd.DataFrame, supplier_summary: pd.DataFrame) -> Dict[str, object]:
    avg_dam = dam["price_eur_mwh"].mean()
    avg_settlement = settlement["settlement_price"].mean()
    settlement_vol_ratio = safe_div(settlement["settlement_price"].std(), dam["price_eur_mwh"].std())

    cheapest_row = profile_insights.loc[profile_insights["best_supplier_cost_eur"].idxmin()]
    highest_use_row = profile_insights.loc[profile_insights["annual_total_kwh"].idxmax()]
    biggest_switch_row = profile_insights.loc[profile_insights["switching_saving_eur"].idxmax()]

    supplier_wins = (
        profile_insights["best_supplier"].dropna().value_counts().rename_axis("provider_name").reset_index(name="wins")
    )
    top_winner = supplier_wins.iloc[0]["provider_name"] if not supplier_wins.empty else "-"

    negative_dam_pct = 100 * dam["price_eur_mwh"].lt(0).mean()
    negative_settlement_pct = 100 * settlement["settlement_price"].lt(0).mean()

    return {
        "avg_dam": avg_dam,
        "avg_settlement": avg_settlement,
        "settlement_vol_ratio": settlement_vol_ratio,
        "cheapest_row": cheapest_row,
        "highest_use_row": highest_use_row,
        "biggest_switch_row": biggest_switch_row,
        "top_winner": top_winner,
        "negative_dam_pct": negative_dam_pct,
        "negative_settlement_pct": negative_settlement_pct,
        "supplier_option_count": supplier_summary["tariff_option_name"].nunique(),
    }



def make_download_file(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dashboard_summary")
    buffer.seek(0)
    return buffer.getvalue()



def add_sidebar_filters(profile_insights: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    st.sidebar.header("Filters")

    profile_options = sorted(profile_insights["profile_name"].dropna().unique().tolist())
    building_options = sorted(profile_insights["building_type"].dropna().unique().tolist())
    heating_options = sorted(profile_insights["heating_system"].dropna().unique().tolist())
    ev_options = sorted(profile_insights["how_many_evs"].dropna().astype(int).unique().tolist())

    selected_profiles = st.sidebar.multiselect("Household profiles", profile_options, default=profile_options)
    selected_buildings = st.sidebar.multiselect("Building type", building_options, default=building_options)
    selected_heating = st.sidebar.multiselect("Heating system", heating_options, default=heating_options)
    selected_evs = st.sidebar.multiselect("EV count", ev_options, default=ev_options)

    filtered = profile_insights[
        profile_insights["profile_name"].isin(selected_profiles)
        & profile_insights["building_type"].isin(selected_buildings)
        & profile_insights["heating_system"].isin(selected_heating)
        & profile_insights["how_many_evs"].fillna(0).astype(int).isin(selected_evs)
    ].copy()

    filters = {
        "profiles": selected_profiles,
        "buildings": selected_buildings,
        "heating": selected_heating,
        "evs": selected_evs,
    }
    return filtered, filters



def display_overview_tab(filtered_profiles: pd.DataFrame, metrics: Dict[str, object], monthly_market: pd.DataFrame, heatmap_df: pd.DataFrame):
    st.subheader("Executive Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg DAM Price", eur(metrics["avg_dam"], 1) + "/MWh")
    c2.metric("Avg Settlement Price", eur(metrics["avg_settlement"], 1) + "/MWh")
    c3.metric("Settlement Volatility vs DAM", f"{num(metrics['settlement_vol_ratio'], 1)}x")
    c4.metric("Top Cheapest Supplier", metrics["top_winner"])
    c5.metric("Profiles in View", f"{filtered_profiles['profile_name'].nunique()}")

    left, right = st.columns([1.1, 1])

    with left:
        market_long = monthly_market.melt(
            id_vars="month",
            value_vars=["avg_dam_price_eur_mwh", "avg_settlement_price_eur_mwh"],
            var_name="series",
            value_name="avg_price_eur_mwh",
        )
        market_long["series"] = market_long["series"].map({
            "avg_dam_price_eur_mwh": "DAM",
            "avg_settlement_price_eur_mwh": "Settlement",
        })
        fig = px.line(
            market_long,
            x="month",
            y="avg_price_eur_mwh",
            color="series",
            markers=True,
            title="Monthly Average Market Price Trend",
        )
        fig.update_layout(height=430, xaxis_title="Month", yaxis_title="€/MWh", legend_title="Series")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        if filtered_profiles.empty:
            st.info("No profiles match the current filters.")
        else:
            display_heatmap = heatmap_df.loc[heatmap_df.index.intersection(filtered_profiles["profile_name"])]
            if not display_heatmap.empty:
                fig_heat = px.imshow(
                    display_heatmap,
                    text_auto=".0f",
                    aspect="auto",
                    color_continuous_scale="RdYlGn_r",
                    title="Supplier Cost Heatmap by Household Profile",
                )
                fig_heat.update_layout(height=430, coloraxis_colorbar_title="€/year")
                st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**What stands out**

- The cheapest supplier across the current view is **{metrics['top_winner']}**.
- Settlement prices are about **{num(metrics['settlement_vol_ratio'], 1)}x** as volatile as DAM on this dataset.
- Negative pricing appears more often in Settlement (**{pct(metrics['negative_settlement_pct'])}**) than DAM (**{pct(metrics['negative_dam_pct'])}**).
- The highest-usage household profile is **{metrics['highest_use_row']['profile_name']}** at **{num(metrics['highest_use_row']['annual_total_kwh'], 0)} kWh/year**.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)



def display_profile_tab(filtered_profiles: pd.DataFrame):
    st.subheader("Household Profile Explorer")
    if filtered_profiles.empty:
        st.warning("No profile data available for the selected filters.")
        return

    breakdown_long = filtered_profiles[[
        "profile_name", "annual_appliances_kwh", "annual_heating_kwh", "annual_ev_kwh"
    ]].melt(
        id_vars="profile_name",
        var_name="load_type",
        value_name="annual_kwh"
    )
    breakdown_long["load_type"] = breakdown_long["load_type"].map({
        "annual_appliances_kwh": "Appliances",
        "annual_heating_kwh": "Heating",
        "annual_ev_kwh": "EV Charging",
    })

    left, right = st.columns(2)
    with left:
        fig_total = px.bar(
            filtered_profiles.sort_values("annual_total_kwh", ascending=True),
            x="annual_total_kwh",
            y="profile_name",
            orientation="h",
            color="heating_system",
            title="Annual Electricity Usage by Household Profile",
            labels={"annual_total_kwh": "Annual kWh", "profile_name": "Profile"},
        )
        fig_total.update_layout(height=450)
        st.plotly_chart(fig_total, use_container_width=True)

    with right:
        fig_breakdown = px.bar(
            breakdown_long,
            x="annual_kwh",
            y="profile_name",
            color="load_type",
            orientation="h",
            barmode="stack",
            title="Consumption Breakdown: Appliances vs Heating vs EV",
            labels={"annual_kwh": "Annual kWh", "profile_name": "Profile", "load_type": "Load Type"},
        )
        fig_breakdown.update_layout(height=450)
        st.plotly_chart(fig_breakdown, use_container_width=True)

    focus_profile = st.selectbox("Select a profile for detailed interpretation", filtered_profiles["profile_name"].tolist())
    p = filtered_profiles.loc[filtered_profiles["profile_name"].eq(focus_profile)].iloc[0]

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Profile interpretation: {focus_profile}**

- Building type: **{p['building_type']}**
- Heating system: **{p['heating_system']}**
- EV count: **{int(p['how_many_evs'])}**
- Total annual demand: **{num(p['annual_total_kwh'], 0)} kWh**
- Dominant load driver: **{p['dominant_load']}**
- Appliances share: **{pct(p['appliances_share_pct'])}**
- Heating share: **{pct(p['heating_share_pct'])}**
- EV share: **{pct(p['ev_share_pct'])}**
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)



def display_tariff_tab(filtered_profiles: pd.DataFrame, provider_profile_cost: pd.DataFrame, supplier_summary: pd.DataFrame):
    st.subheader("Tariff Comparison")
    if filtered_profiles.empty:
        st.warning("No tariff data available for the selected filters.")
        return

    available_profiles = filtered_profiles["profile_name"].tolist()
    selected_profile = st.selectbox("Choose a household profile", available_profiles, key="tariff_profile")

    profile_provider = provider_profile_cost[provider_profile_cost["profile_name"].eq(selected_profile)].sort_values("annual_total_cost_eur")
    profile_ranked = supplier_summary[supplier_summary["profile_name"].eq(selected_profile)].sort_values("annual_total_cost_eur")
    profile_row = filtered_profiles[filtered_profiles["profile_name"].eq(selected_profile)].iloc[0]

    left, right = st.columns([1.2, 1])
    with left:
        fig_provider = px.bar(
            profile_provider,
            x="provider_name",
            y="annual_total_cost_eur",
            color="provider_name",
            title="Annual Cost by Provider for Selected Profile",
            labels={"annual_total_cost_eur": "Annual Cost (€)", "provider_name": "Provider"},
        )
        fig_provider.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig_provider, use_container_width=True)

    with right:
        comparison_df = pd.DataFrame({
            "Measure": ["Cheapest Supplier", "Average Supplier", "Most Expensive Supplier"],
            "Annual Cost (€)": [
                profile_row["cheapest_supplier_cost_eur"],
                profile_row["avg_supplier_cost_eur"],
                profile_row["most_expensive_supplier_cost_eur"],
            ]
        })
        fig_spread = px.bar(
            comparison_df,
            x="Measure",
            y="Annual Cost (€)",
            color="Measure",
            title="Supplier Cost Spread",
        )
        fig_spread.update_layout(height=430, showlegend=False)
        st.plotly_chart(fig_spread, use_container_width=True)

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Tariff insight for this profile**

- Best supplier: **{profile_row['best_supplier']}**
- Best tariff option: **{profile_row['best_tariff']}**
- Best supplier annual cost: **{eur(profile_row['best_supplier_cost_eur'])}**
- Best market benchmark cost: **{eur(profile_row['best_market_cost_eur'])}**
- Retail premium vs best market benchmark: **{eur(profile_row['retail_premium_vs_market_eur'])}**
- Potential switching saving inside supplier tariffs: **{eur(profile_row['switching_saving_eur'])}**
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Ranked tariff options")
    ranked_display = profile_ranked[[
        "rank", "provider_name", "tariff_option_name", "category", "annual_total_cost_eur",
        "difference_vs_dam_10pct_eur", "difference_vs_settlement_10pct_eur", "cheaper_than_market"
    ]].copy()
    ranked_display = ranked_display.rename(columns={
        "provider_name": "Provider",
        "tariff_option_name": "Tariff Option",
        "category": "Category",
        "annual_total_cost_eur": "Annual Cost (€)",
        "difference_vs_dam_10pct_eur": "Diff vs DAM 10% (€)",
        "difference_vs_settlement_10pct_eur": "Diff vs Settlement 10% (€)",
        "cheaper_than_market": "Cheaper Than Market",
        "rank": "Rank",
    })
    st.dataframe(ranked_display, use_container_width=True, hide_index=True)



def display_market_tab(dam: pd.DataFrame, settlement: pd.DataFrame, monthly_market: pd.DataFrame):
    st.subheader("Market Benchmark")

    sample_rate = st.select_slider("Trend detail", options=["Full", "Half", "Quarter"], value="Quarter")
    step_map = {"Full": 1, "Half": 2, "Quarter": 4}
    step = step_map[sample_rate]

    dam_sample = dam[["timestamp", "price_eur_mwh"]].dropna().iloc[::step].copy()
    dam_sample["series"] = "DAM"
    dam_sample = dam_sample.rename(columns={"price_eur_mwh": "price"})

    settlement_sample = settlement[["timestamp", "settlement_price"]].dropna().iloc[::step].copy()
    settlement_sample["series"] = "Settlement"
    settlement_sample = settlement_sample.rename(columns={"settlement_price": "price"})

    trend_df = pd.concat([dam_sample, settlement_sample], ignore_index=True).sort_values("timestamp")

    left, right = st.columns(2)
    with left:
        fig_trend = px.line(
            trend_df,
            x="timestamp",
            y="price",
            color="series",
            title="DAM vs Settlement Price Trend",
            labels={"timestamp": "Timestamp", "price": "€/MWh", "series": "Series"},
        )
        fig_trend.update_layout(height=430)
        st.plotly_chart(fig_trend, use_container_width=True)

    with right:
        settlement_compare = settlement[["timestamp", "settlement_price", "predicted_settlement_price"]].dropna().copy()
        settlement_compare = settlement_compare.iloc[::step]
        long_compare = settlement_compare.melt(
            id_vars="timestamp",
            value_vars=["settlement_price", "predicted_settlement_price"],
            var_name="series",
            value_name="price",
        )
        long_compare["series"] = long_compare["series"].map({
            "settlement_price": "Actual Settlement",
            "predicted_settlement_price": "Predicted Settlement",
        })
        fig_pred = px.line(
            long_compare,
            x="timestamp",
            y="price",
            color="series",
            title="Actual vs Predicted Settlement Price",
            labels={"timestamp": "Timestamp", "price": "€/MWh", "series": "Series"},
        )
        fig_pred.update_layout(height=430)
        st.plotly_chart(fig_pred, use_container_width=True)

    monthly_long = monthly_market.melt(
        id_vars="month",
        value_vars=[
            "avg_dam_price_eur_mwh",
            "avg_settlement_price_eur_mwh",
            "avg_predicted_settlement_price_eur_mwh",
        ],
        var_name="series",
        value_name="avg_price",
    )
    monthly_long["series"] = monthly_long["series"].map({
        "avg_dam_price_eur_mwh": "DAM",
        "avg_settlement_price_eur_mwh": "Settlement",
        "avg_predicted_settlement_price_eur_mwh": "Predicted Settlement",
    })
    fig_monthly = px.bar(
        monthly_long,
        x="month",
        y="avg_price",
        color="series",
        barmode="group",
        title="Monthly Average Market Prices",
        labels={"month": "Month", "avg_price": "€/MWh", "series": "Series"},
    )
    fig_monthly.update_layout(height=420)
    st.plotly_chart(fig_monthly, use_container_width=True)

    corr = settlement[["settlement_price", "predicted_settlement_price"]].dropna().corr().iloc[0, 1]
    mae = np.mean(np.abs(
        settlement["settlement_price"].dropna().values[:len(settlement["predicted_settlement_price"].dropna())]
        - settlement["predicted_settlement_price"].dropna().values[:len(settlement["settlement_price"].dropna())]
    ))

    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(
        f"""
**Market interpretation**

- DAM provides the wholesale baseline, while Settlement shows balancing outcomes after deviations from expected positions.
- The correlation between actual and predicted settlement prices is **{num(corr, 2)}**, which helps you discuss forecast tracking quality.
- The mean absolute error between actual and predicted settlement prices is about **{eur(mae, 1)}/MWh**.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)



def display_recommendation_tab(filtered_profiles: pd.DataFrame):
    st.subheader("Decision Summary & Export")
    if filtered_profiles.empty:
        st.warning("No records available for export.")
        return

    summary_table = filtered_profiles[[
        "profile_name", "building_type", "occupants", "heating_system", "how_many_evs",
        "annual_total_kwh", "dominant_load", "best_supplier", "best_tariff",
        "best_supplier_cost_eur", "best_market_reference", "best_market_option",
        "best_market_cost_eur", "retail_premium_vs_market_eur", "switching_saving_eur"
    ]].copy().sort_values("annual_total_kwh", ascending=False)

    summary_table = summary_table.rename(columns={
        "profile_name": "Profile",
        "building_type": "Building Type",
        "occupants": "Occupants",
        "heating_system": "Heating System",
        "how_many_evs": "EV Count",
        "annual_total_kwh": "Annual kWh",
        "dominant_load": "Dominant Load",
        "best_supplier": "Best Supplier",
        "best_tariff": "Best Tariff",
        "best_supplier_cost_eur": "Best Supplier Cost (€)",
        "best_market_reference": "Best Market Reference",
        "best_market_option": "Best Market Option",
        "best_market_cost_eur": "Best Market Cost (€)",
        "retail_premium_vs_market_eur": "Retail Premium vs Market (€)",
        "switching_saving_eur": "Switching Saving (€)",
    })

    top_switch = summary_table.loc[summary_table["Switching Saving (€)"].idxmax()]
    highest_market_gap = summary_table.loc[summary_table["Retail Premium vs Market (€)"].idxmax()]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(
            f"""
**Biggest switching opportunity**

- Profile: **{top_switch['Profile']}**
- Best supplier: **{top_switch['Best Supplier']}**
- Potential saving vs most expensive supplier: **{eur(top_switch['Switching Saving (€)'])}**
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(
            f"""
**Highest premium above market benchmark**

- Profile: **{highest_market_gap['Profile']}**
- Best supplier cost: **{eur(highest_market_gap['Best Supplier Cost (€)'])}**
- Best market benchmark: **{eur(highest_market_gap['Best Market Cost (€)'])}**
- Premium: **{eur(highest_market_gap['Retail Premium vs Market (€)'])}**
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    export_bytes = make_download_file(summary_table)
    st.download_button(
        label="Download decision summary",
        data=export_bytes,
        file_name="dashboard_decision_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# -----------------------------
# App
# -----------------------------
st.title("⚡ Tariff Intelligence Dashboard")
st.caption("Business-style dashboard for household tariff comparison, market benchmarking, and switching insights.")

with st.sidebar:
    st.markdown("### Upload workbook")
    uploaded_file = st.file_uploader(
        "Upload the Excel file",
        type=["xlsx"],
        help="The workbook must contain All_Profile_Summary, Profile_Overview, DAM, and Settlement sheets.",
    )

if uploaded_file is None:
    st.info("Upload your workbook to generate the dashboard.")
    st.stop()

try:
    data = load_workbook(uploaded_file)
except Exception as e:
    st.error(f"Could not read the workbook: {e}")
    st.stop()

profile_insights = data["profile_insights"]
filtered_profiles, active_filters = add_sidebar_filters(profile_insights)
filtered_profile_names = filtered_profiles["profile_name"].unique().tolist()

filtered_provider_profile_cost = data["provider_profile_cost"][
    data["provider_profile_cost"]["profile_name"].isin(filtered_profile_names)
]

filtered_supplier_summary = data["supplier_summary"][
    data["supplier_summary"]["profile_name"].isin(filtered_profile_names)
]

filtered_heatmap = data["heatmap_df"]

if filtered_profile_names:
    filtered_heatmap = filtered_heatmap.loc[
        filtered_heatmap.index.intersection(filtered_profile_names)
    ]
else:
    filtered_heatmap = filtered_heatmap.iloc[0:0]

metrics = build_overview_metrics(
    profile_insights,
    data["dam"],
    data["settlement"],
    data["supplier_summary"]
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Profile Explorer",
    "Tariff Comparison",
    "Market Benchmark",
    "Decision Summary",
])

with tab1:
    display_overview_tab(filtered_profiles, metrics, data["monthly_market"], filtered_heatmap)

with tab2:
    display_profile_tab(filtered_profiles)

with tab3:
    display_tariff_tab(filtered_profiles, filtered_provider_profile_cost, filtered_supplier_summary)

with tab4:
    display_market_tab(data["dam"], data["settlement"], data["monthly_market"])

with tab5:
    display_recommendation_tab(filtered_profiles)

st.markdown("---")
st.markdown(
    '<div class="small-note">Built to work with future workbooks that follow the same sheet names and column structure.</div>',
    unsafe_allow_html=True,
)
