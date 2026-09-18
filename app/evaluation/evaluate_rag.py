import asyncio
import json
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithoutReference,
    ContextRecall,
    Faithfulness,
)

from app.core.config import get_settings
from app.generation.rag_chain import RAGService


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

settings = get_settings()


# ---------------------------------------------------------
# Initialize OpenAI clients
# ---------------------------------------------------------

openai_client = AsyncOpenAI(
    api_key=settings.openai_api_key, base_url="https://openrouter.ai/api/v1",
)

evaluator_llm = llm_factory(
    settings.llm_model,
    client=openai_client,
)

evaluator_embeddings = embedding_factory(
    "openai",
    model=settings.embedding_model,
    client=openai_client,
)


# ---------------------------------------------------------
# RAGAS metrics
# ---------------------------------------------------------

faithfulness = Faithfulness(
    llm=evaluator_llm
)

answer_relevancy = AnswerRelevancy(
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
)

context_precision = ContextPrecisionWithoutReference(
    llm=evaluator_llm
)

context_recall = ContextRecall(
    llm=evaluator_llm
)


# ---------------------------------------------------------
# Load evaluation dataset
# ---------------------------------------------------------

def load_dataset():

    path = Path(
        "app/evaluation/dataset.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Evaluate one sample
# ---------------------------------------------------------

async def evaluate_sample(
    rag: RAGService,
    sample: dict,
):

    question = sample["question"]

    reference = sample["reference"]

    # -----------------------------------------------------
    # Query RAG
    # -----------------------------------------------------

    rag_result = rag.answer(
        question=question,
        session_id=f"eval-{id(sample)}",
        top_k=5,
    )

    answer = rag_result["answer"]

    # -----------------------------------------------------
    # Extract contexts
    # -----------------------------------------------------

    retrieved_results = (
        rag.retriever.search(
            query=question,
            top_k=5,
        )
    )

    contexts = [
        result["text"]
        for result in retrieved_results
    ]

    # -----------------------------------------------------
    # RAGAS evaluation
    # -----------------------------------------------------

    faithfulness_result = (
        await faithfulness.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
    )

    relevancy_result = (
        await answer_relevancy.ascore(
            user_input=question,
            response=answer,
        )
    )

    precision_result = (
        await context_precision.ascore(
            user_input=question,
            retrieved_contexts=contexts,
        )
    )

    recall_result = (
        await context_recall.ascore(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
    )

    return {
        "question": question,
        "answer": answer,
        "faithfulness": (
            faithfulness_result.value
        ),
        "answer_relevancy": (
            relevancy_result.value
        ),
        "context_precision": (
            precision_result.value
        ),
        "context_recall": (
            recall_result.value
        ),
    }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

async def main():

    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)

    dataset = load_dataset()

    rag = RAGService()

    results = []

    for index, sample in enumerate(
        dataset,
        start=1,
    ):

        print(
            f"\nEvaluating "
            f"{index}/{len(dataset)}..."
        )

        try:

            result = await evaluate_sample(
                rag,
                sample,
            )

            results.append(result)

            print(
                f"Faithfulness: "
                f"{result['faithfulness']:.4f}"
            )

            print(
                f"Answer Relevancy: "
                f"{result['answer_relevancy']:.4f}"
            )

            print(
                f"Context Precision: "
                f"{result['context_precision']:.4f}"
            )

            print(
                f"Context Recall: "
                f"{result['context_recall']:.4f}"
            )

        except Exception as exc:

            print(
                f"Evaluation failed: {exc}"
            )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    if not results:

        print(
            "\nNo evaluation results generated."
        )

        return

    results_directory = Path(
        "evaluation/results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        results
    )

    output_file = (
        results_directory
        / "ragas_results.csv"
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    # -----------------------------------------------------
    # Calculate averages
    # -----------------------------------------------------

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    averages = dataframe[
        metric_columns
    ].mean()

    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    for metric, value in averages.items():

        print(
            f"{metric}: {value:.4f}"
        )

    print(
        f"\nResults saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":

    asyncio.run(main())