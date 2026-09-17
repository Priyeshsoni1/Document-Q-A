from app.generation.rag_chain import RAGService


def main():

    print("=" * 70)
    print("DOCUMENT Q&A")
    print("=" * 70)

    rag = RAGService()

    question = input(
        "\nAsk a question about your documents: "
    ).strip()

    if not question:
        print("Question cannot be empty.")
        return

    result = rag.answer(
        question=question,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    if not result["citations"]:
        print("No sources found.")
        return

    for citation in result["citations"]:
        print(
            f"- {citation['source']} "
            f"(Page {citation['page']}) "
            f"[Score: {citation['score']:.4f}]"
        )


if __name__ == "__main__":
    main()