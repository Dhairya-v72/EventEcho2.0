# EventEcho – Quantitative Event Study & Market Reaction Analysis

An interactive financial market intelligence platform that studies how stock prices and volatility behave around corporate earnings announcements.

EventEcho combines quantitative event-study methods, statistical validation, and NLP-based financial news analysis in a multi-page Streamlit dashboard.

## Project Overview

Financial announcements can influence stock prices, returns, and market volatility. However, observing a price movement after an announcement does not automatically mean that the announcement caused it.

EventEcho addresses this problem by comparing actual stock returns with benchmark-based expected returns and analysing abnormal returns around selected corporate events.

The platform also provides an NLP module to explore financial news sentiment, topic patterns, and named entities.

## Key Features

### 1. Market Intelligence Overview
- Overall event-study summary.
- Total eligible events and companies.
- Average and median cumulative abnormal returns.
- Company-level performance overview.
- Market-wide volatility comparison.

### 2. Event Study
- Market-model expected returns.
- Abnormal Return (AR) calculation.
- Cumulative Abnormal Return (CAR).
- Average Abnormal Return (AAR).
- Cumulative Average Abnormal Return (CAAR).
- Event-time analysis.

### 3. Market Analysis
- Historical closing prices.
- Daily stock returns.
- Normalized price-performance comparison.
- S&P 500 benchmark comparison.
- Company-level market statistics.

### 4. Company Comparison
- Average CAR by company.
- Positive and negative event reactions.
- Pre-event and post-event volatility.
- Market-model beta and R-squared comparison.

### 5. NLP Intelligence
- FinBERT sentiment classification.
- BERTopic-based topic discovery.
- MiniLM sentence embeddings and similarity analysis.
- spaCy named-entity recognition.
- Sentiment probability and entropy analysis.

### 6. Statistical Validation
- One-sample t-tests.
- Wilcoxon signed-rank tests.
- Alternative event-window analysis.
- Volatility significance testing.
- Market-model diagnostics.
- Pipeline validation checks.

### 7. Strategy & Backtesting
- Exploratory event-driven trading simulation.
- Day-zero abnormal-return direction signal.
- Day-plus-one abnormal-return payoff.
- Transaction-cost assumptions.
- Company-wise simulated performance.

## Methodology

### Quantitative Event Study

The quantitative event-study module uses historical closing prices and S&P 500 benchmark returns.

The market model is:

R_it = α_i + β_i R_mt + ε_it

Where:

- R_it = Return of company i on day t.
- R_mt = Benchmark market return.
- α_i = Intercept.
- β_i = Market sensitivity.
- ε_it = Residual return.

Expected return:

Expected Return = α_i + β_i × R_mt

Abnormal return:

AR_it = Actual Return - Expected Return

Cumulative abnormal return:

CAR_i = Σ AR_it

The project uses an estimation window of (-120, -21), a gap period of (-20, -11), and an event window of (-10, +10).

### NLP Analysis

The NLP pipeline explores financial news headlines using:

- FinBERT for financial sentiment classification.
- BERTopic for exploratory topic discovery.
- MiniLM for semantic embeddings and similarity.
- spaCy for named-entity extraction.

The NLP dataset and the verified earnings-event dataset are analysed separately because the generic news headlines have not been directly matched to the verified earnings events.

## Dataset and Analysis

The expanded quantitative analysis covers:

- 8 selected companies.
- 52 eligible earnings events.
- 1,092 abnormal-return observations.
- 21 event-time observations for AAR and CAAR.
- 50 unique financial news headlines for the NLP analysis.

Selected companies:

| Ticker | Company |
|---|---|
| AAPL | Apple |
| MSFT | Microsoft |
| TSLA | Tesla |
| NVDA | NVIDIA |
| AMZN | Amazon |
| GOOGL | Alphabet |
| META | Meta Platforms |
| JPM | JPMorgan Chase |

## Key Findings

- The average CAR across 52 eligible events was approximately +1.47%.
- The overall CAR result was not statistically significant at the 5% level.
- Post-event volatility was higher than pre-event volatility in the expanded sample.
- The paired volatility tests indicated a statistically significant increase.
- The NLP analysis identified sentiment and topic patterns, but these remain exploratory.
- The NLP headlines were not directly mapped to the verified earnings events.
- The exploratory trading simulation should not be interpreted as proof of a profitable trading strategy.

## Technology Stack

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Financial Data:** yfinance

**Machine Learning and NLP:** Scikit-learn, FinBERT, BERTopic, Sentence Transformers, MiniLM, spaCy

**Statistical Analysis:** SciPy, Statsmodels

**Visualization:** Streamlit, Matplotlib, Plotly

**Development Tools:** Jupyter Notebook, VS Code, Git, GitHub

## Project Structure

```text
EventEcho/
│
├── app.py
├── pages/
│   ├── 2_Event_Study.py
│   ├── 3_Market_Analysis.py
│   ├── 4_Company_Comparison.py
│   ├── 5_NLP_Intelligence.py
│   ├── 6_Statistical_Validation.py
│   └── 7_Strategy_Backtesting.py
│
├── modules/
│   ├── data_loader.py
│   ├── event_analysis.py
│   ├── nlp_analysis.py
│   └── ui.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── notebooks/
├── models/
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd EventEcho
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Limitations

- The quantitative sample contains 52 eligible events across eight companies.
- Event dates and announcement timing require careful interpretation; some historical timing details are provisional.
- The news dataset contains repeated headline patterns and is not equivalent to a fully verified real-time news feed.
- NLP sentiment predictions are not independently validated against manually labelled ground truth.
- NLP news records are not directly linked to the verified earnings events.
- Statistical tests may be affected by dependence between events occurring near the same dates.
- The strategy simulation does not model actual execution prices, slippage, liquidity, taxes, or short-selling constraints.
- Event-study associations do not establish causality.

## Future Improvements

- Expand the event database to additional companies and sectors.
- Integrate verified historical news and announcement timestamps.
- Improve NLP-to-event matching.
- Add more robust cross-sectional statistical methods.
- Evaluate models on a larger unseen sample.
- Incorporate realistic execution costs and trading constraints.
- Add interactive event and company-level report downloads.

## Disclaimer

EventEcho is an academic and research-oriented financial analysis project. It is not investment advice, and its outputs should not be used as direct trading recommendations.

---

**EventEcho – Understanding Market Reactions Through Data, Statistics and NLP.**