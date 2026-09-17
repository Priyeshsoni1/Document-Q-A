from app.generation.rag_chain import RAGService


def main():

    print("=" * 70)
    print("CONVERSATIONAL DOCUMENT Q&A")
    print("=" * 70)

    rag = RAGService()

    session_id = "test-session-001"

    while True:

        question = input(
            "\nYou: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit",
        }:
            break

        if not question:
            continue

        result = rag.answer(
            question=question,
            session_id=session_id,
            top_k=5,
        )

        print(
            f"\nAssistant: {result['answer']}"
        )

        if result["citations"]:

            print("\nSources:")

            for citation in result[
                "citations"
            ]:

                print(
                    f"- {citation['source']} "
                    f"(Page {citation['page']})"
                )


if __name__ == "__main__":
    main()