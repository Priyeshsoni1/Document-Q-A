from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="RAGAS Evaluation",
    page_icon="📊",
    layout="wide",
)


st.title("📊 RAGAS Evaluation Dashboard")

results_file = Path(
    "evaluation/results/ragas_results.csv"
)


if not results_file.exists():

    st.warning(
        "No evaluation results found. "
        "Run evaluation/evaluate_rag.py first."
    )

    st.stop()


df = pd.read_csv(
    results_file
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

metrics = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


averages = df[
    metrics
].mean()


# ---------------------------------------------------------
# Metric cards
# ---------------------------------------------------------

columns = st.columns(4)


for column, metric in zip(
    columns,
    metrics,
):

    column.metric(
        metric.replace(
            "_",
            " ",
        ).title(),
        f"{averages[metric]:.3f}",
    )


# ---------------------------------------------------------
# Target
# ---------------------------------------------------------

overall_score = averages.mean()

st.divider()

st.subheader(
    "Overall RAG Quality"
)

st.metric(
    "Average Score",
    f"{overall_score:.3f}",
)


if overall_score >= 0.75:

    st.success(
        "Target threshold ≥ 0.75 achieved."
    )

else:

    st.warning(
        "Target threshold ≥ 0.75 not achieved."
    )


# ---------------------------------------------------------
# Chart
# ---------------------------------------------------------

st.subheader(
    "Metric Comparison"
)

chart_data = (
    averages
    .rename(
        lambda x: x.replace(
            "_",
            " ",
        ).title()
    )
)

st.bar_chart(
    chart_data
)


# ---------------------------------------------------------
# Detailed results
# ---------------------------------------------------------

st.subheader(
    "Evaluation Samples"
)

st.dataframe(
    df,
    use_container_width=True,
)