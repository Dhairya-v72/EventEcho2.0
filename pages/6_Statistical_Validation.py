
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.ui import (
    apply_global_styles,
    render_sidebar,
    render_page_header,
    render_footer
)

st.set_page_config(
    page_title="Statistical Validation | EventEcho",
    layout="wide"
)

apply_global_styles()
render_sidebar()


@st.cache_data
def load_validation_data():
    return {
        "significance": load_dataset("expanded_significance"),
        "validation": load_dataset("expanded_validation"),
        "volatility": load_dataset("expanded_volatility"),
        "advanced": load_dataset("advanced_model_validation"),
        "events": load_dataset("expanded_event_car_summary")
    }


try:
    data = load_validation_data()

    significance = data["significance"].copy()
    validation = data["validation"].copy()
    volatility = data["volatility"].copy()
    advanced = data["advanced"].copy()
    events = data["events"].copy()

    render_page_header(
        "Statistical Validation",
        "Evaluate the statistical significance, robustness and "
        "reliability of the expanded event-study results."
    )

    st.divider()

    # Overall CAR test
    st.subheader("Overall Abnormal Return Significance")

    car_mean = events["CAR"].mean()
    car_median = events["CAR"].median()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Eligible Events", len(events))
    c2.metric("Mean CAR", f"{car_mean:+.3f}%")
    c3.metric("Median CAR", f"{car_median:+.3f}%")
    c4.metric(
        "Positive CAR Events",
        f'{(events["CAR"] > 0).sum()} / {len(events)}'
    )

    st.caption(
        "The expanded sample contains 52 eligible earnings events "
        "across eight companies."
    )

    st.divider()

    # Event window tests
    st.subheader("Alternative Event Window Tests")

    st.dataframe(
        significance.round(4),
        use_container_width=True,
        hide_index=True
    )

    if not significance.empty:
        chart_data = significance.set_index("window")[
            ["mean_CAR"]
        ]

        st.markdown("**Mean CAR across event windows**")
        st.bar_chart(chart_data)

        p_threshold = st.number_input(
            "Significance threshold",
            min_value=0.001,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.3f"
        )

        st.markdown("**Statistical interpretation**")

        for _, row in significance.iterrows():
            t_p = row["t_pvalue"]
            w_p = row["wilcoxon_pvalue"]

            t_result = (
                "statistically significant"
                if pd.notna(t_p) and t_p < p_threshold
                else "not statistically significant"
            )

            w_result = (
                "statistically significant"
                if pd.notna(w_p) and w_p < p_threshold
                else "not statistically significant"
            )

            st.write(
                f'**{row["window"]}:** t-test is {t_result} '
                f'(p = {t_p:.4f}); Wilcoxon test is {w_result} '
                f'(p = {w_p:.4f}).'
            )

    st.divider()

    # Volatility test
    st.subheader("Pre-event vs Post-event Volatility")

    pre_mean = volatility["pre_volatility"].mean()
    post_mean = volatility["post_volatility"].mean()
    change = volatility["volatility_change"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric("Pre-event Volatility", f"{pre_mean:.3f}%")
    c2.metric("Post-event Volatility", f"{post_mean:.3f}%")
    c3.metric(
        "Average Change",
        f"{change:+.3f} pp",
        delta=f"{change:+.3f} percentage points"
    )

    st.bar_chart(
        pd.DataFrame({
            "Volatility": [pre_mean, post_mean]
        }, index=["Pre-event", "Post-event"])
    )

    st.markdown(
        "The paired volatility tests from the expanded analysis "
        "show an increase in post-event volatility."
    )

    st.write(
        "**Paired t-test p-value:** 0.0127"
    )

    st.write(
        "**Wilcoxon signed-rank p-value:** 0.0014"
    )

    st.caption(
        "These results indicate an association between the event period "
        "and volatility changes. They do not establish that announcements "
        "caused the changes."
    )

    st.divider()

    # Validation report
    st.subheader("Pipeline Validation Checks")

    if not validation.empty:
        passed_count = validation["passed"].astype(bool).sum()
        total_checks = len(validation)

        c1, c2 = st.columns(2)

        c1.metric("Passed Checks", passed_count)
        c2.metric("Total Checks", total_checks)

        st.dataframe(
            validation,
            use_container_width=True,
            hide_index=True
        )

        if passed_count == total_checks:
            st.success("All recorded validation checks passed.")
        else:
            st.warning("Some validation checks did not pass.")

    st.divider()

    # Advanced validation
    st.subheader("Advanced Model Validation")

    if not advanced.empty:
        st.dataframe(
            advanced,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Advanced validation results are unavailable.")

    with st.expander("How to interpret these tests"):
        st.markdown(
            """
            - **t-test:** Tests whether the mean CAR differs from zero
              under its assumptions.
            - **Wilcoxon signed-rank test:** A non-parametric test of
              paired differences against zero.
            - **Event window sensitivity:** Checks whether conclusions
              change when the event window changes.
            - **Volatility comparison:** Compares volatility before and
              after the event.
            - **R-squared:** Measures how much variation the market model
              explains within its estimation sample.
            - **p-value:** A measure of compatibility between the observed
              result and the null hypothesis; it is not the probability
              that the null hypothesis is true.
            """
        )

    st.warning(
        "The 52 event observations may still share market-wide shocks, "
        "so independence assumptions in simple tests should be treated "
        "with caution. Statistical significance does not imply causality."
    )

    render_footer()

except Exception as e:
    st.error("Unable to load Statistical Validation.")
    st.exception(e)
