
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.ui import (
    apply_global_styles,
    render_sidebar,
    render_page_header,
    render_footer
)

st.set_page_config(page_title="Company Comparison | EventEcho", layout="wide")
apply_global_styles()
render_sidebar()


@st.cache_data
def load_data():
    return {
        "events": load_dataset("expanded_event_car_summary"),
        "volatility": load_dataset("expanded_volatility"),
        "models": load_dataset("expanded_market_models"),
        "companies": load_dataset("expanded_company_list")
    }


try:
    data = load_data()

    events = data["events"].copy()
    volatility = data["volatility"].copy()
    models = data["models"].copy()
    companies = data["companies"].copy()

    render_page_header(
        "Company Comparison",
        "Compare abnormal returns, volatility and market-model "
        "characteristics across the eight selected companies."
    )

    st.divider()

    # Filters
    st.sidebar.subheader("Comparison Filters")

    available = sorted(events["company"].dropna().unique())

    selected = st.sidebar.multiselect(
        "Select Companies",
        available,
        default=available
    )

    events = events[events["company"].isin(selected)]
    volatility = volatility[volatility["company"].isin(selected)]
    models = models[models["company"].isin(selected)]

    if events.empty:
        st.warning("No data available for the selected companies.")
        st.stop()

    # Company performance
    summary = (
        events.groupby("company")
        .agg(
            Total_Events=("event_id", "count"),
            Average_CAR=("CAR", "mean"),
            Median_CAR=("CAR", "median"),
            Positive_Events=("CAR", lambda x: (x > 0).sum()),
            Negative_Events=("CAR", lambda x: (x < 0).sum())
        )
        .reset_index()
        .sort_values("Average_CAR", ascending=False)
    )

    st.subheader("Company Performance Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Companies Analysed", events["company"].nunique())
    c2.metric("Eligible Events", len(events))
    c3.metric(
        "Highest Average CAR",
        f'{summary.iloc[0]["Average_CAR"]:+.2f}%'
    )

    st.divider()

    st.subheader("Average CAR by Company")
    st.bar_chart(
        summary.set_index("company")[["Average_CAR"]],
        use_container_width=True
    )

    st.caption(
        "CAR is the cumulative abnormal return over the selected event "
        "window. Company event counts differ, so interpret averages carefully."
    )

    st.subheader("Positive vs Negative Event Reactions")
    st.bar_chart(
        summary.set_index("company")[
            ["Positive_Events", "Negative_Events"]
        ],
        use_container_width=True
    )

    st.subheader("Company Performance Table")
    st.dataframe(
        summary.round(3),
        use_container_width=True,
        hide_index=True
    )

    # Volatility
    st.divider()
    st.subheader("Company-wise Volatility Comparison")

    if not volatility.empty:
        vol_summary = (
            volatility.groupby("company")
            .agg(
                Pre_Event=("pre_volatility", "mean"),
                Post_Event=("post_volatility", "mean"),
                Volatility_Change=("volatility_change", "mean"),
                Change_Percent=("volatility_change_percent", "mean")
            )
            .reset_index()
            .round(3)
        )

        left, right = st.columns(2)

        with left:
            st.markdown("**Pre-event volatility**")
            st.bar_chart(
                vol_summary.set_index("company")[["Pre_Event"]]
            )

        with right:
            st.markdown("**Post-event volatility**")
            st.bar_chart(
                vol_summary.set_index("company")[["Post_Event"]]
            )

        st.markdown("**Average volatility change**")
        st.bar_chart(
            vol_summary.set_index("company")[["Volatility_Change"]]
        )

        st.dataframe(
            vol_summary,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Volatility data is unavailable.")

    # Market model
    st.divider()
    st.subheader("Market Model Comparison")

    if not models.empty:
        model_summary = (
            models.groupby("company")
            .agg(
                Average_Beta=("beta", "mean"),
                Average_R2=("r_squared", "mean"),
                Average_Alpha=("alpha", "mean"),
                Number_of_Models=("event_id", "count")
            )
            .reset_index()
            .round(4)
        )

        left, right = st.columns(2)

        with left:
            st.markdown("**Average market-model beta**")
            st.bar_chart(
                model_summary.set_index("company")[["Average_Beta"]]
            )

        with right:
            st.markdown("**Average model R-squared**")
            st.bar_chart(
                model_summary.set_index("company")[["Average_R2"]]
            )

        st.dataframe(
            model_summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Beta measures estimated sensitivity to the S&P 500. "
            "R-squared describes fit within the estimation window; "
            "it is not a measure of out-of-sample predictive accuracy."
        )

    # Event-level table
    st.divider()
    st.subheader("Event-level Comparison")

    st.dataframe(
        events.sort_values("CAR", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    render_footer()

except Exception as e:
    st.error("Unable to load Company Comparison.")
    st.exception(e)
