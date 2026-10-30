
from pathlib import Path
import pandas as pd
import numpy as np


# =====================================================
# PROJECT DIRECTORIES
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"


# =====================================================
# DATASET REGISTRY
# =====================================================

DATASETS = {

    # ---------------- RAW DATA ----------------
    "raw_news": RAW_DIR / "financial_news_events.csv",

    # ---------------- PILOT DATA ----------------
    "pilot_closing_prices": PROCESSED_DIR / "pilot_closing_prices.csv",
    "pilot_daily_returns": PROCESSED_DIR / "pilot_daily_returns.csv",

    # ---------------- EXPANDED MARKET DATA ----------------
    "expanded_closing_prices": PROCESSED_DIR / "expanded_closing_prices.csv",
    "expanded_daily_returns": PROCESSED_DIR / "expanded_daily_returns.csv",
    "expanded_company_list": PROCESSED_DIR / "expanded_company_list.csv",

    # ---------------- EVENT DATA ----------------
    "eligible_events": PROCESSED_DIR / "eligible_events.csv",
    "expanded_event_candidates": PROCESSED_DIR / "expanded_event_candidates.csv",
    "event_window_eligibility": PROCESSED_DIR / "event_window_eligibility.csv",

    # ---------------- EXPANDED QUANTITATIVE RESULTS ----------------
    "expanded_event_car_summary": RESULTS_DIR / "expanded_event_car_summary.csv",
    "expanded_abnormal_returns": RESULTS_DIR / "expanded_abnormal_returns.csv",
    "expanded_aar_caar": RESULTS_DIR / "expanded_aar_caar_results.csv",
    "expanded_significance": RESULTS_DIR / "expanded_significance_tests.csv",
    "expanded_volatility": RESULTS_DIR / "expanded_volatility_results.csv",
    "expanded_market_models": RESULTS_DIR / "expanded_market_model_results.csv",
    "expanded_validation": RESULTS_DIR / "expanded_validation_report.csv",

    # ---------------- PILOT RESULTS ----------------
    "pilot_abnormal_returns": RESULTS_DIR / "abnormal_returns.csv",
    "pilot_event_car_summary": RESULTS_DIR / "event_car_summary.csv",
    "pilot_aar_caar": RESULTS_DIR / "aar_caar_results.csv",
    "pilot_event_study_summary": RESULTS_DIR / "final_event_study_summary.csv",
    "pilot_market_models": RESULTS_DIR / "market_model_results.csv",
    "pilot_volatility": RESULTS_DIR / "volatility_results.csv",
    "pilot_volatility_comparison": RESULTS_DIR / "volatility_comparison.csv",
    "pilot_t_test": RESULTS_DIR / "t_test_results.csv",
    "pilot_wilcoxon": RESULTS_DIR / "wilcoxon_results.csv",
    "pilot_event_window_matrix": RESULTS_DIR / "event_window_car_matrix.csv",

    # Keep the original pilot key
    "pilot_advanced_validation": RESULTS_DIR / "advanced_model_validation.csv",

    # Alias used by the Statistical Validation dashboard
    "advanced_model_validation": RESULTS_DIR / "advanced_model_validation.csv",

    # ---------------- NLP DATA ----------------
    "nlp_features": PROCESSED_DIR / "integrated_nlp_features.csv",
    "finbert_results": PROCESSED_DIR / "finbert_sentiment_results.csv",
    "bertopic_assignments": PROCESSED_DIR / "bertopic_topic_assignments.csv",
    "bertopic_info": PROCESSED_DIR / "bertopic_topic_info.csv",
    "unique_headlines": PROCESSED_DIR / "unique_nlp_headlines.csv",
    "minilm_headline_mapping": PROCESSED_DIR / "minilm_headline_mapping.csv",
    "spacy_filtered_entities": PROCESSED_DIR / "spacy_filtered_entities.csv",
    "spacy_raw_entities": PROCESSED_DIR / "spacy_raw_entities.csv",

    # ---------------- NLP + PILOT INTEGRATION ----------------
    "integrated_event_data": RESULTS_DIR / "integrated_event_sentiment_car.csv",
    "company_sentiment_comparison": RESULTS_DIR / "company_sentiment_car_comparison.csv",
    "event_news_mapping": RESULTS_DIR / "final_event_news_mapping.csv",
    "event_news_mapping_template": RESULTS_DIR / "event_news_mapping_template.csv",
    "company_car_comparison": RESULTS_DIR / "company_car_comparison.csv",

    # ---------------- AUDIT ----------------
    "initial_data_audit": RESULTS_DIR / "initial_data_audit.csv",
}


# =====================================================
# NUMPY ARRAY REGISTRY
# =====================================================

ARRAYS = {
    "minilm_embeddings": PROCESSED_DIR / "minilm_embeddings.npy",
    "minilm_similarity_matrix": PROCESSED_DIR / "minilm_similarity_matrix.npy",
}


# =====================================================
# LOAD CSV DATASET
# =====================================================

def load_dataset(name):
    """
    Load a registered CSV dataset.

    Parameters
    ----------
    name : str
        Registered dataset key.

    Returns
    -------
    pandas.DataFrame
    """

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")

    file_path = DATASETS[name]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at: {file_path}"
        )

    df = pd.read_csv(file_path)

    # Convert common date columns safely
    date_columns = [
        "Date",
        "event_date",
        "reaction_date"
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df


# =====================================================
# LOAD NUMPY ARRAY
# =====================================================

def load_array(name):
    """
    Load a registered NumPy array.
    """

    if name not in ARRAYS:
        raise ValueError(f"Unknown array: {name}")

    file_path = ARRAYS[name]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Array '{name}' not found at: {file_path}"
        )

    return np.load(file_path)


# =====================================================
# LOAD ALL REGISTERED DATASETS
# =====================================================

def load_all_datasets():
    """
    Load all available registered CSV datasets.

    Missing files are skipped and reported.
    """

    datasets = {}

    for name, file_path in DATASETS.items():

        if file_path.exists():
            datasets[name] = load_dataset(name)

        else:
            print(f"Warning: Missing dataset - {name}")

    return datasets
