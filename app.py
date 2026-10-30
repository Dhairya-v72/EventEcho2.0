
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.event_analysis import (
    get_expanded_event_summary,
    get_volatility_summary
)

from modules.ui import (
    apply_global_styles,
    render_sidebar,
    render_page_header,
    render_footer
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="EventEcho | Market Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# GLOBAL UI
# =========================================================

apply_global_styles()
render_sidebar()


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_dashboard_data():

    return {
        "events": load_dataset("expanded_event_car_summary"),
        "eligible_events": load_dataset("eligible_events"),
        "aar_caar": load_dataset("expanded_aar_caar"),
        "volatility": load_dataset("expanded_volatility"),
        "significance": load_dataset("expanded_significance"),
        "validation": load_dataset("expanded_validation")
    }


# =========================================================
# PAGE HEADER
# =========================================================

render_page_header(
    "Market Intelligence Overview",
    "Explore how corporate announcements are associated with "
    "stock market performance, abnormal returns and volatility. "
    "EventEcho combines quantitative event-study methods with "
    "financial news intelligence and statistical validation."
)


# =========================================================
# MAIN DASHBOARD
# =========================================================

try:

    # -----------------------------------------------------
    # Load datasets
    # -----------------------------------------------------

    data = load_dashboard_data()

    events = data["events"].copy()
    eligible_events = data["eligible_events"].copy()
    aar_caar = data["aar_caar"].copy()
    volatility = data["volatility"].copy()
    significance = data["significance"].copy()
    validation = data["validation"].copy()

    # Convert dates
    for df in [events, eligible_events]:

        for col in ["event_date", "reaction_date"]:

            if col in df.columns:

                df[col] = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )


    # -----------------------------------------------------
    # GLOBAL FILTERS
    # -----------------------------------------------------

    st.sidebar.subheader("Dashboard Filters")

    available_companies = sorted(
        events["company"].dropna().unique()
    )

    selected_companies = st.sidebar.multiselect(
        "Companies",
        options=available_companies,
        default=available_companies,
        key="overview_companies"
    )

    available_dates = events["event_date"].dropna()

    selected_dates = None

    if not available_dates.empty:

        min_date = available_dates.min().date()
        max_date = available_dates.max().date()

        selected_dates = st.sidebar.date_input(
            "Event Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="overview_dates"
        )


    # -----------------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------------

    filtered_events = events[
        events["company"].isin(selected_companies)
    ].copy()

    if (
        isinstance(selected_dates, (tuple, list))
        and len(selected_dates) == 2
    ):

        start_date, end_date = selected_dates

        filtered_events = filtered_events[
            filtered_events["event_date"].dt.date.between(
                start_date,
                end_date
            )
        ]

    selected_ids = filtered_events["event_id"].unique()

    filtered_volatility = volatility[
        volatility["event_id"].isin(selected_ids)
    ].copy()

    filtered_eligible = eligible_events[
        eligible_events["event_id"].isin(selected_ids)
    ].copy()


    # =====================================================
    # MARKET OVERVIEW
    # =====================================================

    st.divider()

    st.subheader("Market Overview")

    if filtered_events.empty:

        st.warning(
            "No events found. Please adjust the dashboard filters."
        )

    else:

        summary = get_expanded_event_summary(filtered_events)

        total_events = summary["total_events"]

        positive_ratio = (
            summary["positive_events"] / total_events * 100
        )

        negative_ratio = (
            summary["negative_events"] / total_events * 100
        )

        # First row
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Events",
            total_events
        )

        col2.metric(
            "Companies",
            summary["companies"]
        )

        col3.metric(
            "Positive CAR",
            summary["positive_events"]
        )

        col4.metric(
            "Negative CAR",
            summary["negative_events"]
        )

        # Second row
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Average CAR",
            f'{summary["average_CAR"]:+.2f}%'
        )

        col2.metric(
            "Median CAR",
            f'{summary["median_CAR"]:+.2f}%'
        )

        col3.metric(
            "Positive Event Ratio",
            f"{positive_ratio:.1f}%"
        )

        col4.metric(
            "Negative Event Ratio",
            f"{negative_ratio:.1f}%"
        )


    # =====================================================
    # EVENT REACTION OVERVIEW
    # =====================================================

    st.divider()

    st.subheader("Event Reaction Overview")

    st.markdown(
        '<div class="ee-caption">'
        'Explore event-wise CAR distribution and company-level '
        'market reactions.'
        '</div>',
        unsafe_allow_html=True
    )

    if not filtered_events.empty:

        left, right = st.columns(2)

        with left:

            st.markdown("#### Event-wise CAR")

            car_data = (
                filtered_events[
                    ["event_id", "CAR"]
                ]
                .sort_values("CAR")
                .set_index("event_id")
            )

            st.bar_chart(
                car_data,
                use_container_width=True
            )

        with right:

            st.markdown("#### Company-wise Average CAR")

            company_car = (
                filtered_events.groupby("company")["CAR"]
                .mean()
                .sort_values(ascending=False)
            )

            st.bar_chart(
                company_car,
                use_container_width=True
            )


    # =====================================================
    # AAR AND CAAR ANALYSIS
    # =====================================================

    st.divider()

    st.subheader("Market Reaction Around Events")

    st.markdown(
        '<div class="ee-caption">'
        'Average and cumulative average abnormal returns '
        'across the event window.'
        '</div>',
        unsafe_allow_html=True
    )

    if not aar_caar.empty:

        aar_caar = aar_caar.sort_values("event_time")

        left, right = st.columns(2)

        with left:

            st.markdown("#### Average Abnormal Return (AAR)")

            st.line_chart(
                aar_caar.set_index("event_time")[["AAR"]],
                use_container_width=True
            )

        with right:

            st.markdown(
                "#### Cumulative Average Abnormal Return (CAAR)"
            )

            st.line_chart(
                aar_caar.set_index("event_time")[["CAAR"]],
                use_container_width=True
            )

        st.caption(
            "These AAR and CAAR results represent the complete "
            "expanded sample of 52 eligible events. They are not "
            "recalculated using the sidebar filters."
        )


    # =====================================================
    # VOLATILITY ANALYSIS
    # =====================================================

    st.divider()

    st.subheader("Volatility Analysis")

    st.markdown(
        '<div class="ee-caption">'
        'Compare stock volatility before and after corporate events.'
        '</div>',
        unsafe_allow_html=True
    )

    if not filtered_volatility.empty:

        vol_summary = get_volatility_summary(
            filtered_volatility
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Pre-event Volatility",
            f'{vol_summary["average_pre_volatility"]:.2f}%'
        )

        col2.metric(
            "Post-event Volatility",
            f'{vol_summary["average_post_volatility"]:.2f}%'
        )

        col3.metric(
            "Average Volatility Change",
            f'{vol_summary["average_volatility_change"]:+.2f} pp'
        )

        vol_chart = filtered_volatility[
            [
                "event_id",
                "pre_volatility",
                "post_volatility"
            ]
        ].set_index("event_id")

        st.bar_chart(
            vol_chart,
            use_container_width=True
        )

        st.caption(
            f'{vol_summary["events_with_increased_volatility"]} '
            "events showed increased volatility, while "
            f'{vol_summary["events_with_decreased_volatility"]} '
            "events showed decreased volatility."
        )

    else:

        st.info("No volatility records available for the selected filters.")


    # =====================================================
    # STATISTICAL EVIDENCE
    # =====================================================

    st.divider()

    st.subheader("Statistical Evidence")

    st.markdown(
        """
        Evaluate whether cumulative abnormal returns differ
        significantly from zero across alternative event windows.
        """
    )

    if not significance.empty:

        st.dataframe(
            significance,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "The overall CAR is not statistically significant "
            "at the 5% level in the expanded sample. Observed "
            "market reactions should not be interpreted as "
            "proof of causality."
        )

        st.caption(
            "Significance results represent the complete 52-event "
            "sample and are not recalculated using sidebar filters."
        )


    # =====================================================
    # EVENT EXPLORER
    # =====================================================

    st.divider()

    st.subheader("Event Explorer")

    if not filtered_eligible.empty:

        display_columns = [
            col for col in [
                "event_id",
                "company",
                "company_name",
                "event_date",
                "reaction_date",
                "event_category",
                "headline",
                "announcement_timing",
                "verification_status"
            ]
            if col in filtered_eligible.columns
        ]

        st.dataframe(
            filtered_eligible[display_columns],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No event records available for the selected filters."
        )


    # =====================================================
    # DATA VALIDATION
    # =====================================================

    with st.expander("Dataset Validation Report"):

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # FOOTER
    # =====================================================

    render_footer()


except Exception as e:

    st.error("Unable to load the dashboard.")

    st.exception(e)
