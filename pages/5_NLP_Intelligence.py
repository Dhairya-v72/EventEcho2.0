
import streamlit as st
import pandas as pd

from modules.data_loader import load_dataset
from modules.ui import (
    apply_global_styles,
    render_sidebar,
    render_page_header,
    render_footer
)

st.set_page_config(page_title="NLP Intelligence | EventEcho", layout="wide")
apply_global_styles()
render_sidebar()


@st.cache_data
def load_nlp_data():
    return {
        "features": load_dataset("nlp_features"),
        "finbert": load_dataset("finbert_results"),
        "topics": load_dataset("bertopic_assignments"),
        "topic_info": load_dataset("bertopic_info"),
        "entities": load_dataset("spacy_filtered_entities")
    }


try:
    data = load_nlp_data()

    features = data["features"].copy()
    finbert = data["finbert"].copy()
    topics = data["topics"].copy()
    topic_info = data["topic_info"].copy()
    entities = data["entities"].copy()

    render_page_header(
        "NLP Intelligence",
        "Explore financial headline sentiment, topic discovery and "
        "named-entity extraction using NLP models."
    )

    st.divider()

    # Dataset overview
    st.subheader("NLP Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Unique Headlines", len(finbert))
    c2.metric("Negative Headlines", (finbert["FinBERT_Sentiment"] == "negative").sum())
    c3.metric("Positive Headlines", (finbert["FinBERT_Sentiment"] == "positive").sum())
    c4.metric("Neutral Headlines", (finbert["FinBERT_Sentiment"] == "neutral").sum())

    st.caption(
        "NLP analysis uses 50 unique headlines extracted from the news dataset, "
        "not all 3,024 original records."
    )

    st.divider()

    # FinBERT
    st.subheader("FinBERT Sentiment Analysis")

    sentiment_counts = (
        finbert["FinBERT_Sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .to_frame("Headline_Count")
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Sentiment distribution**")
        st.bar_chart(sentiment_counts)

    with right:
        st.markdown("**Average sentiment probabilities**")

        probability_cols = [
            "Positive_Probability",
            "Negative_Probability",
            "Neutral_Probability"
        ]

        probability_summary = (
            finbert[probability_cols]
            .mean()
            .rename_axis("Probability")
            .to_frame("Average")
        )

        probability_summary.index = [
            "Positive", "Negative", "Neutral"
        ]

        st.bar_chart(probability_summary)

    st.markdown("**Sentiment confidence and entropy**")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Average Confidence",
        f'{finbert["FinBERT_Confidence"].mean():.3f}'
    )

    c2.metric(
        "Average Entropy",
        f'{finbert["Sentiment_Entropy"].mean():.3f}'
    )

    c3.metric(
        "Median Entropy",
        f'{finbert["Sentiment_Entropy"].median():.3f}'
    )

    st.caption(
        "Higher entropy indicates a more uncertain or distributed "
        "sentiment prediction. Confidence and entropy are model outputs, "
        "not verified ground-truth sentiment."
    )

    st.divider()

    # BERTopic
    st.subheader("BERTopic - Financial News Topics")

    if not topics.empty:
        topic_counts = (
            topics["BERTopic_Name"]
            .value_counts()
            .rename_axis("Topic")
            .to_frame("Headline_Count")
        )

        st.bar_chart(topic_counts)

        st.markdown("**Topic information**")

        topic_columns = [
            col for col in [
                "Topic", "Count", "Name",
                "Representation", "Representative_Docs"
            ]
            if col in topic_info.columns
        ]

        st.dataframe(
            topic_info[topic_columns],
            use_container_width=True,
            hide_index=True
        )

        noise_count = (
            (topics["BERTopic_ID"] == -1).sum()
            if "BERTopic_ID" in topics.columns else 0
        )

        st.info(
            f"{noise_count} headlines were assigned to the BERTopic "
            "noise group. These records did not receive a regular topic."
        )

    st.divider()

    # Topic probability
    st.subheader("Topic Assignment Confidence")

    if "Topic_Probability" in topics.columns:
        st.metric(
            "Average Topic Probability",
            f'{topics["Topic_Probability"].mean():.3f}'
        )

        st.bar_chart(
            topics.groupby("BERTopic_Name")["Topic_Probability"].mean()
        )

    st.divider()

    # NER
    st.subheader("Named Entity Recognition")

    if not entities.empty:
        entity_columns = [
            col for col in [
                "Headline", "Entity", "Entity_Label",
                "Label", "Start", "End"
            ]
            if col in entities.columns
        ]

        st.metric(
            "Extracted Entity Records",
            len(entities)
        )

        st.dataframe(
            entities[entity_columns] if entity_columns else entities,
            use_container_width=True,
            hide_index=True
        )

        label_column = next(
            (
                col for col in ["Entity_Label", "Label"]
                if col in entities.columns
            ),
            None
        )

        if label_column:
            st.markdown("**Entity categories**")
            st.bar_chart(entities[label_column].value_counts())

    else:
        st.info("No filtered entities were found.")

    st.divider()

    # Headline explorer
    st.subheader("Headline Sentiment Explorer")

    sentiment_filter = st.selectbox(
        "Filter by sentiment",
        ["All", "positive", "negative", "neutral"]
    )

    search_text = st.text_input("Search headline")

    headline_view = finbert.copy()

    if sentiment_filter != "All":
        headline_view = headline_view[
            headline_view["FinBERT_Sentiment"] == sentiment_filter
        ]

    if search_text:
        headline_view = headline_view[
            headline_view["Headline"].str.contains(
                search_text, case=False, na=False
            )
        ]

    st.dataframe(
        headline_view,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "Important: these NLP features are exploratory. The 50 generic "
        "headlines have not been directly matched to the 52 verified "
        "earnings events. Do not interpret NLP sentiment and market CAR "
        "as a unified event-level causal relationship."
    )

    render_footer()

except Exception as e:
    st.error("Unable to load NLP Intelligence.")
    st.exception(e)
