
import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_dataset
from modules.ui import (
    apply_global_styles,
    render_sidebar,
    render_page_header,
    render_footer
)

st.set_page_config(
    page_title="Strategy & Backtesting | EventEcho",
    layout="wide"
)

apply_global_styles()
render_sidebar()


@st.cache_data
def load_strategy_data():
    return {
        "abnormal_returns": load_dataset("expanded_abnormal_returns"),
        "events": load_dataset("expanded_event_car_summary"),
        "significance": load_dataset("expanded_significance")
    }


try:
    data = load_strategy_data()

    abnormal_returns = data["abnormal_returns"].copy()
    events = data["events"].copy()
    significance = data["significance"].copy()

    render_page_header(
        "Strategy & Backtesting",
        "Explore a simple event-driven trading hypothesis using "
        "abnormal returns from the expanded event-study sample."
    )

    st.divider()

    st.subheader("Strategy Configuration")

    col1, col2 = st.columns(2)

    with col1:
        transaction_cost = st.slider(
            "Assumed round-trip transaction cost (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        )

    with col2:
        signal_type = st.selectbox(
            "Trading signal",
            [
                "Day 0 direction",
                "Always long",
                "Always short"
            ]
        )

    st.caption(
        "The cost is an assumed percentage-point deduction per trade. "
        "It is a user-configurable modelling assumption, not a measured "
        "brokerage or execution cost."
    )

    st.divider()

    # Prepare event-time returns
    day_zero = abnormal_returns[
        abnormal_returns["event_time"] == 0
    ][["event_id", "company", "abnormal_return"]].copy()

    day_one = abnormal_returns[
        abnormal_returns["event_time"] == 1
    ][["event_id", "abnormal_return"]].copy()

    day_zero = day_zero.rename(
        columns={"abnormal_return": "Day_0_AR"}
    )

    day_one = day_one.rename(
        columns={"abnormal_return": "Day_1_AR"}
    )

    trades = day_zero.merge(
        day_one,
        on="event_id",
        how="inner"
    )

    trades = trades.dropna(
        subset=["Day_0_AR", "Day_1_AR"]
    ).copy()

    # Generate positions
    if signal_type == "Day 0 direction":
        trades["Position"] = np.sign(trades["Day_0_AR"])

    elif signal_type == "Always long":
        trades["Position"] = 1

    else:
        trades["Position"] = -1

    trades = trades[trades["Position"] != 0].copy()

    # Hypothetical next-day abnormal-return payoff
    trades["Gross_Return"] = (
        trades["Position"] * trades["Day_1_AR"]
    )

    trades["Transaction_Cost"] = transaction_cost

    trades["Net_Return"] = (
        trades["Gross_Return"] - trades["Transaction_Cost"]
    )

    if trades.empty:
        st.warning("No eligible trades are available for this strategy.")
        st.stop()

    # Summary
    total_trades = len(trades)
    average_gross = trades["Gross_Return"].mean()
    average_net = trades["Net_Return"].mean()
    total_net = trades["Net_Return"].sum()

    win_rate = (
        (trades["Net_Return"] > 0).mean() * 100
    )

    positive_trades = (
        trades["Net_Return"] > 0
    ).sum()

    negative_trades = (
        trades["Net_Return"] < 0
    ).sum()

    st.subheader("Backtesting Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Trades", total_trades)

    c2.metric(
        "Average Gross Return",
        f"{average_gross:+.3f}%"
    )

    c3.metric(
        "Average Net Return",
        f"{average_net:+.3f}%"
    )

    c4.metric(
        "Win Rate",
        f"{win_rate:.2f}%"
    )

    st.divider()

    # Return distribution
    st.subheader("Trade Return Distribution")

    left, right = st.columns(2)

    with left:
        st.markdown("**Gross vs Net Average Return**")

        st.bar_chart(
            pd.DataFrame(
                {
                    "Return": [
                        average_gross,
                        average_net
                    ]
                },
                index=["Gross", "Net"]
            )
        )

    with right:
        st.markdown("**Winning vs Losing Trades**")

        st.bar_chart(
            pd.DataFrame(
                {
                    "Trades": [
                        positive_trades,
                        negative_trades
                    ]
                },
                index=["Winning", "Losing"]
            )
        )

    st.divider()

    # Company-wise strategy performance
    st.subheader("Company-wise Strategy Performance")

    company_performance = (
        trades.groupby("company")
        .agg(
            Trades=("event_id", "count"),
            Average_Gross_Return=("Gross_Return", "mean"),
            Average_Net_Return=("Net_Return", "mean"),
            Winning_Trades=(
                "Net_Return",
                lambda x: (x > 0).sum()
            )
        )
        .reset_index()
        .sort_values("Average_Net_Return", ascending=False)
    )

    st.bar_chart(
        company_performance.set_index("company")[
            ["Average_Net_Return"]
        ]
    )

    st.dataframe(
        company_performance.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Trade explorer
    st.subheader("Individual Trade Results")

    display_trades = trades[
        [
            "event_id",
            "company",
            "Day_0_AR",
            "Day_1_AR",
            "Position",
            "Gross_Return",
            "Transaction_Cost",
            "Net_Return"
        ]
    ].copy()

    display_trades = display_trades.sort_values(
        "Net_Return",
        ascending=False
    )

    st.dataframe(
        display_trades.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Methodology
    st.subheader("Strategy Methodology")

    st.markdown(
        """
        1. **Signal generation:** Use the direction of day 0 abnormal
           return as the trading signal.
        2. **Position:** Positive day 0 abnormal return creates a long
           position; negative creates a short position.
        3. **Holding period:** The hypothetical position is evaluated
           using day +1 abnormal return.
        4. **Gross return:** Position multiplied by day +1 abnormal return.
        5. **Net return:** Gross return minus the assumed transaction cost.
        """
    )

    with st.expander("Limitations and assumptions"):
        st.markdown(
            """
            - This is an exploratory simulation, not a live trading system.
            - Abnormal returns are benchmark-adjusted statistical quantities,
              not directly investable stock returns.
            - The signal assumes day 0 information is available before
              the day +1 holding-period return.
            - Execution prices, bid-ask spreads, slippage, liquidity,
              taxes and short-selling constraints are not modelled.
            - The sample contains only 52 eligible events.
            - Events from different companies can still share market-wide
              shocks.
            - Results should not be interpreted as evidence of a
              profitable trading strategy.
            """
        )

    st.warning(
        "A positive simulated return does not establish that the strategy "
        "would be profitable in real markets. Further testing on a separate "
        "unseen sample and realistic execution data is necessary."
    )

    render_footer()

except Exception as e:
    st.error("Unable to load Strategy & Backtesting.")
    st.exception(e)
