
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.event_analysis import get_abnormal_return_summary


st.set_page_config(
    page_title="Event Study | EventEcho",
    layout="wide"
)


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #080D19;
}
[data-testid="stMetric"] {
    background: #111B2E;
    border: 1px solid #263247;
    border-radius: 10px;
    padding: 16px;
}
.section-caption {
    color: #94A3B8;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_event_data():
    return {
        "car": load_dataset("expanded_event_car_summary"),
        "ar": load_dataset("expanded_abnormal_returns"),
        "aar_caar": load_dataset("expanded_aar_caar"),
        "significance": load_dataset("expanded_significance"),
        "events": load_dataset("eligible_events")
    }


try:

    data = load_event_data()

    car_df = data["car"].copy()
    ar_df = data["ar"].copy()
    aar_df = data["aar_caar"].copy()
    sig_df = data["significance"].copy()
    events_df = data["events"].copy()

    for df in [car_df, events_df]:
        for col in ["event_date", "reaction_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")


    # =================================================
    # PAGE HEADER
    # =================================================

    st.title("Event Study Analysis")

    st.markdown(
        '<div class="section-caption">'
        'Measure stock market reactions around corporate announcements '
        'using abnormal returns and cumulative abnormal returns.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()


    # =================================================
    # FILTERS
    # =================================================

    st.sidebar.subheader("Event Study Filters")

    companies = sorted(car_df["company"].dropna().unique())

    selected_companies = st.sidebar.multiselect(
        "Companies",
        companies,
        default=companies,
        key="event_study_companies"
    )

    event_ids = sorted(
        car_df[
            car_df["company"].isin(selected_companies)
        ]["event_id"].unique()
    )

    selected_event = st.selectbox(
        "Select an individual event",
        ["All Events"] + event_ids
    )

    filtered_car = car_df[
        car_df["company"].isin(selected_companies)
    ].copy()

    filtered_ar = ar_df[
        ar_df["company"].isin(selected_companies)
    ].copy()

    if selected_event != "All Events":
        filtered_car = filtered_car[
            filtered_car["event_id"] == selected_event
        ]
        filtered_ar = filtered_ar[
            filtered_ar["event_id"] == selected_event
        ]


    # =================================================
    # KPI METRICS
    # =================================================

    st.subheader("Event Study Summary")

    if filtered_car.empty:

        st.warning("No events found for the selected filters.")

    else:

        total = len(filtered_car)
        avg_car = filtered_car["CAR"].mean()
        median_car = filtered_car["CAR"].median()
        positive = (filtered_car["CAR"] > 0).sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Events Analysed", total)
        c2.metric("Average CAR", f"{avg_car:+.2f}%")
        c3.metric("Median CAR", f"{median_car:+.2f}%")
        c4.metric("Positive CAR Events", positive)

    st.divider()


    # =================================================
    # EVENT WINDOW SELECTION
    # =================================================

    st.subheader("Cumulative Abnormal Return by Event Window")

    window_options = {
        "[-1, +1]": "[-1,+1]",
        "[0, +1]": "[0,+1]",
        "[-5, +5]": "[-5,+5]",
        "[-10, +10]": "[-10,+10]"
    }

    selected_window_label = st.selectbox(
        "Select event window",
        list(window_options.keys())
    )

    selected_window = window_options[selected_window_label]

    window_result = sig_df[
        sig_df["window"] == selected_window
    ]

    if not window_result.empty:

        result = window_result.iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean CAR",
            f'{result["mean_CAR"]:+.2f}%'
        )

        c2.metric(
            "T-test p-value",
            f'{result["t_pvalue"]:.4f}'
        )

        c3.metric(
            "Wilcoxon p-value",
            f'{result["wilcoxon_pvalue"]:.4f}'
        )

        if result["t_pvalue"] < 0.05:
            st.success(
                "The t-test indicates statistical significance "
                "at the 5% level."
            )
        else:
            st.info(
                "The t-test does not indicate statistical significance "
                "at the 5% level."
            )

        st.caption(
            "Window significance values are precomputed for the complete "
            "52-event sample. They are not recalculated for sidebar filters."
        )

    st.divider()


    # =================================================
    # ABNORMAL RETURNS
    # =================================================

    st.subheader("Abnormal Return Analysis")

    if not filtered_ar.empty:

        if selected_event == "All Events":

            ar_summary = get_abnormal_return_summary(filtered_ar)

            st.markdown("#### Average Abnormal Return by Event Time")

            st.line_chart(
                ar_summary.set_index("event_time")[["Average_AR"]],
                use_container_width=True
            )

            st.markdown("#### Average AR Distribution")

            st.dataframe(
                ar_summary,
                use_container_width=True,
                hide_index=True
            )

        else:

            event_ar = filtered_ar.sort_values("event_time")

            st.markdown(
                f"#### Abnormal Returns — {selected_event}"
            )

            st.line_chart(
                event_ar.set_index("event_time")[["abnormal_return"]],
                use_container_width=True
            )

            st.dataframe(
                event_ar,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info("No abnormal return observations available.")


    st.divider()


    # =================================================
    # AAR AND CAAR
    # =================================================

    st.subheader("Average and Cumulative Average Abnormal Returns")

    st.markdown(
        '<div class="section-caption">'
        'Full-sample event-time profile across all 52 eligible events.'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("#### AAR")

        st.line_chart(
            aar_df.set_index("event_time")[["AAR"]],
            use_container_width=True
        )

    with right:

        st.markdown("#### CAAR")

        st.line_chart(
            aar_df.set_index("event_time")[["CAAR"]],
            use_container_width=True
        )

    st.divider()


    # =================================================
    # EVENT CAR COMPARISON
    # =================================================

    st.subheader("Event-wise CAR Comparison")

    if not filtered_car.empty:

        chart_data = (
            filtered_car
            .sort_values("CAR")
            .set_index("event_id")[["CAR"]]
        )

        st.bar_chart(
            chart_data,
            use_container_width=True
        )

        st.dataframe(
            filtered_car.sort_values("CAR", ascending=False),
            use_container_width=True,
            hide_index=True
        )


    # =================================================
    # EVENT INFORMATION
    # =================================================

    st.divider()

    st.subheader("Announcement Information")

    event_details = events_df[
        events_df["event_id"].isin(filtered_car["event_id"])
    ].copy()

    if selected_event != "All Events":
        event_details = event_details[
            event_details["event_id"] == selected_event
        ]

    columns = [
        "event_id",
        "company",
        "event_date",
        "reaction_date",
        "event_category",
        "headline",
        "announcement_timing",
        "verification_status"
    ]

    columns = [
        col for col in columns if col in event_details.columns
    ]

    st.dataframe(
        event_details[columns],
        use_container_width=True,
        hide_index=True
    )


    # =================================================
    # FOOTER
    # =================================================

    st.divider()

    st.caption(
        "EventEcho | Event Study Analysis | "
        "Historical association does not establish causality."
    )


except Exception as e:

    st.error("Unable to load Event Study page.")
    st.exception(e)
