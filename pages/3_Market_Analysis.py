
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
    page_title="Market Analysis | EventEcho",
    layout="wide"
)

apply_global_styles()
render_sidebar()


@st.cache_data
def load_market_data():
    return {
        "prices": load_dataset("expanded_closing_prices"),
        "returns": load_dataset("expanded_daily_returns"),
        "companies": load_dataset("expanded_company_list")
    }


try:

    data = load_market_data()

    prices = data["prices"].copy()
    returns = data["returns"].copy()
    companies = data["companies"].copy()

    prices["Date"] = pd.to_datetime(prices["Date"])
    returns["Date"] = pd.to_datetime(returns["Date"])

    # -------------------------------------------------
    # HEADER
    # -------------------------------------------------

    render_page_header(
        "Market Analysis",
        "Explore historical stock prices, daily returns and "
        "market benchmark movements across eight selected companies."
    )

    st.divider()

    # -------------------------------------------------
    # FILTERS
    # -------------------------------------------------

    st.sidebar.subheader("Market Filters")

    stock_columns = [
        col for col in prices.columns if col != "Date"
    ]

    selected_stocks = st.sidebar.multiselect(
        "Select Stocks",
        stock_columns,
        default=stock_columns
    )

    min_date = prices["Date"].min().date()
    max_date = prices["Date"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:

        start_date, end_date = date_range

        filtered_prices = prices[
            prices["Date"].dt.date.between(start_date, end_date)
        ].copy()

        filtered_returns = returns[
            returns["Date"].dt.date.between(start_date, end_date)
        ].copy()

    else:
        filtered_prices = prices.copy()
        filtered_returns = returns.copy()

    # Keep benchmark available for comparison
    chart_stocks = [
        stock for stock in selected_stocks if stock != "SP500"
    ]

    # -------------------------------------------------
    # MARKET SNAPSHOT
    # -------------------------------------------------

    st.subheader("Market Snapshot")

    if filtered_prices.empty or not selected_stocks:

        st.warning("Select at least one stock and a valid date range.")

    else:

        latest = filtered_prices.iloc[-1]

        cols = st.columns(min(len(selected_stocks), 4))

        for i, stock in enumerate(selected_stocks):

            with cols[i % len(cols)]:

                st.metric(
                    stock,
                    f"{latest[stock]:,.2f}"
                    if pd.notna(latest[stock]) else "N/A"
                )

        st.caption(
            f"Latest available trading date: "
            f"{filtered_prices['Date'].max().date()}"
        )

    st.divider()

    # -------------------------------------------------
    # CLOSING PRICE TRENDS
    # -------------------------------------------------

    st.subheader("Historical Closing Prices")

    if not filtered_prices.empty and selected_stocks:

        price_chart = filtered_prices.set_index("Date")[
            selected_stocks
        ]

        st.line_chart(
            price_chart,
            use_container_width=True
        )

        st.caption(
            "Closing prices are shown in their original currency units. "
            "Raw prices are not directly comparable across companies."
        )

    st.divider()

    # -------------------------------------------------
    # NORMALIZED PRICE PERFORMANCE
    # -------------------------------------------------

    st.subheader("Normalized Price Performance")

    st.markdown(
        "Compare relative price movements by setting each selected "
        "stock's first available observation to 100."
    )

    if not filtered_prices.empty and selected_stocks:

        normalized = filtered_prices.set_index("Date")[
            selected_stocks
        ].copy()

        normalized = normalized.dropna(how="all")

        if not normalized.empty:

            normalized = normalized.apply(
                lambda col: (
                    col / col.dropna().iloc[0] * 100
                    if col.notna().any()
                    and col.dropna().iloc[0] != 0
                    else col
                )
            )

            st.line_chart(
                normalized,
                use_container_width=True
            )

    st.divider()

    # -------------------------------------------------
    # DAILY RETURNS
    # -------------------------------------------------

    st.subheader("Daily Returns")

    if not filtered_returns.empty and selected_stocks:

        return_data = filtered_returns.set_index("Date")[
            selected_stocks
        ]

        st.line_chart(
            return_data,
            use_container_width=True
        )

        st.caption(
            "Daily returns are represented in percentage units "
            "as stored in the processed dataset."
        )

    st.divider()

    # -------------------------------------------------
    # BENCHMARK COMPARISON
    # -------------------------------------------------

    st.subheader("Stock vs Market Benchmark")

    if (
        "SP500" in filtered_returns.columns
        and chart_stocks
        and not filtered_returns.empty
    ):

        benchmark_data = filtered_returns[
            ["SP500"] + chart_stocks
        ].set_index(filtered_returns["Date"])

        st.line_chart(
            benchmark_data,
            use_container_width=True
        )

        st.caption(
            "S&P 500 is used as the broad market benchmark. "
            "This comparison is descriptive and does not replace "
            "the market-model expected-return analysis."
        )

    else:

        st.info(
            "Select at least one company stock to compare "
            "against the S&P 500 benchmark."
        )

    st.divider()

    # -------------------------------------------------
    # RETURN STATISTICS
    # -------------------------------------------------

    st.subheader("Return Statistics")

    if not filtered_returns.empty and selected_stocks:

        stats = filtered_returns[selected_stocks].agg(
            ["mean", "median", "std", "min", "max"]
        ).T

        stats.columns = [
            "Mean Return",
            "Median Return",
            "Volatility (Std)",
            "Minimum Return",
            "Maximum Return"
        ]

        st.dataframe(
            stats,
            use_container_width=True
        )

    st.divider()

    # -------------------------------------------------
    # COMPANY INFORMATION
    # -------------------------------------------------

    with st.expander("Company and Sector Information"):

        st.dataframe(
            companies,
            use_container_width=True,
            hide_index=True
        )

    render_footer()


except Exception as e:

    st.error("Unable to load Market Analysis.")

    st.exception(e)
