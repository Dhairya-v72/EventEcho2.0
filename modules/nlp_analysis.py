
import pandas as pd
import numpy as np


# =====================================================
# 1. SENTIMENT DISTRIBUTION
# =====================================================

def get_sentiment_distribution(df):
    """Count headlines belonging to each sentiment category."""

    return (
        df["FinBERT_Sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Event_Count")
    )


# =====================================================
# 2. AVERAGE SENTIMENT CONFIDENCE
# =====================================================

def get_average_sentiment_confidence(df):
    """Calculate average model confidence for each sentiment."""

    return (
        df.groupby("FinBERT_Sentiment")["FinBERT_Confidence"]
        .mean()
        .reset_index(name="Average_Confidence")
    )


# =====================================================
# 3. SENTIMENT-CAR COMPARISON (PILOT DATA)
# =====================================================

def get_sentiment_car_comparison(df):
    """
    Compare average CAR across sentiment categories.

    Use only when sentiment and CAR are available
    in the same verified event-level dataset.
    """

    return (
        df.groupby("FinBERT_Sentiment")["total_CAR"]
        .mean()
        .reset_index(name="Average_CAR")
    )


# =====================================================
# 4. SENTIMENT PROBABILITY SUMMARY
# =====================================================

def get_sentiment_probability_summary(df):
    """Calculate average positive, negative and neutral probabilities."""

    probability_columns = [
        "Positive_Probability",
        "Negative_Probability",
        "Neutral_Probability"
    ]

    available_columns = [
        col for col in probability_columns if col in df.columns
    ]

    if not available_columns:
        return pd.DataFrame()

    return (
        df[available_columns]
        .mean()
        .rename_axis("Sentiment")
        .reset_index(name="Average_Probability")
    )


# =====================================================
# 5. SENTIMENT ENTROPY SUMMARY
# =====================================================

def get_entropy_summary(df):
    """Summarize uncertainty in FinBERT sentiment predictions."""

    if "Sentiment_Entropy" not in df.columns:
        return {}

    entropy = df["Sentiment_Entropy"].dropna()

    if entropy.empty:
        return {}

    return {
        "average_entropy": entropy.mean(),
        "median_entropy": entropy.median(),
        "minimum_entropy": entropy.min(),
        "maximum_entropy": entropy.max()
    }


# =====================================================
# 6. TOPIC DISTRIBUTION
# =====================================================

def get_topic_distribution(df):
    """Count headlines assigned to each BERTopic topic."""

    if "BERTopic_Name" not in df.columns:
        raise ValueError("BERTopic_Name column is missing.")

    return (
        df["BERTopic_Name"]
        .fillna("Unassigned")
        .value_counts()
        .rename_axis("Topic")
        .reset_index(name="Headline_Count")
    )


# =====================================================
# 7. TOPIC-WISE SENTIMENT SUMMARY
# =====================================================

def get_topic_sentiment_summary(df):
    """Compare sentiment across BERTopic categories."""

    required_columns = [
        "BERTopic_Name",
        "FinBERT_Sentiment"
    ]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("Required topic or sentiment columns are missing.")

    return (
        df.groupby(
            ["BERTopic_Name", "FinBERT_Sentiment"],
            dropna=False
        )
        .size()
        .reset_index(name="Headline_Count")
        .sort_values(
            ["BERTopic_Name", "Headline_Count"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )


# =====================================================
# 8. TOPIC INFORMATION
# =====================================================

def get_topic_information(df):
    """Return BERTopic metadata and topic descriptions."""

    columns = [
        "Topic",
        "Count",
        "Name",
        "Representation",
        "Representative_Docs"
    ]

    available_columns = [
        col for col in columns if col in df.columns
    ]

    return df[available_columns].copy()


# =====================================================
# 9. ENTITY EXTRACTION SUMMARY
# =====================================================

def get_entity_summary(df):
    """Summarize extracted entities and their labels."""

    if "Entity" not in df.columns:
        return pd.DataFrame()

    if "Entity_Label" in df.columns:
        return (
            df.groupby(
                ["Entity", "Entity_Label"],
                dropna=False
            )
            .size()
            .reset_index(name="Occurrences")
            .sort_values(
                "Occurrences",
                ascending=False
            )
            .reset_index(drop=True)
        )

    return (
        df["Entity"]
        .value_counts()
        .rename_axis("Entity")
        .reset_index(name="Occurrences")
    )


# =====================================================
# 10. ENTITY COUNT SUMMARY FROM INTEGRATED FEATURES
# =====================================================

def get_entity_count_summary(df):
    """Summarize the number of named entities per headline."""

    if "NER_Entity_Count" not in df.columns:
        return {}

    counts = df["NER_Entity_Count"].dropna()

    if counts.empty:
        return {}

    return {
        "total_headlines": len(df),
        "average_entities": counts.mean(),
        "maximum_entities": counts.max(),
        "headlines_with_entities": (counts > 0).sum()
    }


# =====================================================
# 11. TOPIC PROBABILITY SUMMARY
# =====================================================

def get_topic_probability_summary(df):
    """Summarize BERTopic assignment probability by topic."""

    if not all(
        col in df.columns
        for col in ["BERTopic_Name", "Topic_Probability"]
    ):
        return pd.DataFrame()

    return (
        df.groupby("BERTopic_Name")["Topic_Probability"]
        .agg(
            Average_Probability="mean",
            Median_Probability="median",
            Headline_Count="count"
        )
        .reset_index()
        .sort_values(
            "Average_Probability",
            ascending=False
        )
        .reset_index(drop=True)
    )


# =====================================================
# 12. HEADLINE SIMILARITY
# =====================================================

def get_headline_similarity(similarity_matrix, headline_mapping):
    """
    Convert MiniLM similarity matrix into a readable
    pairwise headline similarity table.

    similarity_matrix: NumPy array (n x n)
    headline_mapping: DataFrame containing Headline column
    """

    matrix = np.asarray(similarity_matrix)

    headlines = headline_mapping["Headline"].tolist()

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Similarity matrix must be square.")

    if matrix.shape[0] != len(headlines):
        raise ValueError(
            "Similarity matrix size must match headline count."
        )

    pairs = []

    for i in range(len(headlines)):
        for j in range(i + 1, len(headlines)):

            pairs.append({
                "Headline_1": headlines[i],
                "Headline_2": headlines[j],
                "Similarity": matrix[i, j]
            })

    return (
        pd.DataFrame(pairs)
        .sort_values("Similarity", ascending=False)
        .reset_index(drop=True)
    )
