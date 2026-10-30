
import pandas as pd


# =====================================================
# 1. PILOT EVENT ANALYSIS
# =====================================================

def filter_events(
    df,
    company=None,
    sentiment=None,
    car_direction=None,
    event_id=None
):
    """Filter integrated pilot event data."""

    filtered_df = df.copy()

    if company is not None:
        companies = company if isinstance(company, list) else [company]
        filtered_df = filtered_df[
            filtered_df["company"].isin(companies)
        ]

    if sentiment is not None:
        sentiments = sentiment if isinstance(sentiment, list) else [sentiment]
        filtered_df = filtered_df[
            filtered_df["FinBERT_Sentiment"].isin(sentiments)
        ]

    if car_direction is not None:
        directions = (
            car_direction
            if isinstance(car_direction, list)
            else [car_direction]
        )
        filtered_df = filtered_df[
            filtered_df["CAR_Direction"].isin(directions)
        ]

    if event_id is not None:
        event_ids = event_id if isinstance(event_id, list) else [event_id]
        filtered_df = filtered_df[
            filtered_df["event_id"].isin(event_ids)
        ]

    return filtered_df.reset_index(drop=True)


def get_event_summary(df):
    """Generate basic statistics for pilot integrated event data."""

    return {
        "total_events": len(df),
        "companies": df["company"].nunique(),
        "average_CAR": df["total_CAR"].mean(),
        "positive_events": (df["total_CAR"] > 0).sum(),
        "negative_events": (df["total_CAR"] < 0).sum()
    }


# =====================================================
# 2. EXPANDED EVENT SUMMARY
# =====================================================

def get_expanded_event_summary(df):
    """
    Generate overall statistics for the expanded
    quantitative event-study dataset.

    Expected columns:
    event_id, company, CAR, positive_CAR
    """

    return {
        "total_events": len(df),
        "companies": df["company"].nunique(),
        "average_CAR": df["CAR"].mean(),
        "median_CAR": df["CAR"].median(),
        "positive_events": (df["CAR"] > 0).sum(),
        "negative_events": (df["CAR"] < 0).sum(),
        "zero_CAR_events": (df["CAR"] == 0).sum()
    }


# =====================================================
# 3. COMPANY-WISE CAR COMPARISON
# =====================================================

def get_company_car_comparison(df):
    """
    Calculate company-wise CAR statistics.

    Expected columns:
    company, CAR
    """

    comparison = (
        df.groupby("company")
        .agg(
            Total_Events=("event_id", "count"),
            Average_CAR=("CAR", "mean"),
            Median_CAR=("CAR", "median"),
            Positive_Events=("CAR", lambda x: (x > 0).sum()),
            Negative_Events=("CAR", lambda x: (x < 0).sum())
        )
        .reset_index()
    )

    return comparison.sort_values(
        "Average_CAR",
        ascending=False
    ).reset_index(drop=True)


# =====================================================
# 4. EVENT-WINDOW ANALYSIS
# =====================================================

def get_event_window_summary(df):
    """
    Return CAR statistics for different event windows.

    Expected columns:
    window, events, mean_CAR, median_CAR,
    positive_events, negative_events
    """

    columns = [
        "window",
        "events",
        "mean_CAR",
        "median_CAR",
        "positive_events",
        "negative_events"
    ]

    return df[columns].copy()


# =====================================================
# 5. STATISTICAL SIGNIFICANCE SUMMARY
# =====================================================

def get_significance_summary(df):
    """
    Return statistical test results for different
    event windows.

    Expected columns:
    window, t_statistic, t_pvalue,
    wilcoxon_statistic, wilcoxon_pvalue
    """

    columns = [
        "window",
        "events",
        "mean_CAR",
        "median_CAR",
        "t_statistic",
        "t_pvalue",
        "wilcoxon_statistic",
        "wilcoxon_pvalue"
    ]

    return df[columns].copy()


# =====================================================
# 6. VOLATILITY ANALYSIS
# =====================================================

def get_volatility_summary(df):
    """
    Calculate overall pre-event and post-event
    volatility statistics.

    Expected columns:
    pre_volatility, post_volatility,
    volatility_change, volatility_change_percent
    """

    return {
        "average_pre_volatility": df["pre_volatility"].mean(),
        "average_post_volatility": df["post_volatility"].mean(),
        "average_volatility_change": df["volatility_change"].mean(),
        "average_volatility_change_percent": (
            df["volatility_change_percent"].mean()
        ),
        "events_with_increased_volatility": (
            df["volatility_change"] > 0
        ).sum(),
        "events_with_decreased_volatility": (
            df["volatility_change"] < 0
        ).sum()
    }


def get_company_volatility_comparison(df):
    """Compare volatility changes across companies."""

    comparison = (
        df.groupby("company")
        .agg(
            Total_Events=("event_id", "count"),
            Average_Pre_Volatility=("pre_volatility", "mean"),
            Average_Post_Volatility=("post_volatility", "mean"),
            Average_Volatility_Change=("volatility_change", "mean"),
            Average_Change_Percent=("volatility_change_percent", "mean")
        )
        .reset_index()
    )

    return comparison.sort_values(
        "Average_Volatility_Change",
        ascending=False
    ).reset_index(drop=True)


# =====================================================
# 7. ABNORMAL RETURN ANALYSIS
# =====================================================

def get_abnormal_return_summary(df):
    """
    Calculate average abnormal returns for each event-time day.

    Expected columns:
    event_time, abnormal_return
    """

    summary = (
        df.groupby("event_time")
        .agg(
            Average_AR=("abnormal_return", "mean"),
            Median_AR=("abnormal_return", "median"),
            Std_AR=("abnormal_return", "std"),
            Event_Count=("abnormal_return", "count")
        )
        .reset_index()
        .sort_values("event_time")
    )

    return summary.reset_index(drop=True)


# =====================================================
# 8. EVENT-LEVEL ABNORMAL RETURN ANALYSIS
# =====================================================

def get_event_abnormal_returns(df, event_id=None):
    """Get abnormal returns for selected events."""

    filtered_df = df.copy()

    if event_id is not None:
        event_ids = (
            event_id if isinstance(event_id, list) else [event_id]
        )

        filtered_df = filtered_df[
            filtered_df["event_id"].isin(event_ids)
        ]

    return filtered_df.sort_values(
        ["event_id", "event_time"]
    ).reset_index(drop=True)


# =====================================================
# 9. MARKET MODEL SUMMARY
# =====================================================

def get_market_model_summary(df):
    """
    Summarize market-model parameters.

    Expected columns:
    event_id, company, alpha, beta,
    r_squared, estimation_observations
    """

    return {
        "total_models": len(df),
        "average_alpha": df["alpha"].mean(),
        "average_beta": df["beta"].mean(),
        "average_r_squared": df["r_squared"].mean(),
        "average_estimation_observations": (
            df["estimation_observations"].mean()
        )
    }
