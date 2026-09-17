from app.retrieval.retriever import Retriever


def print_results(results):
    print("\n" + "=" * 70)
    print(f"RETRIEVED RESULTS: {len(results)}")
    print("=" * 70)

    for index, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nResult #{index}")
        print("-" * 70)

        print(f"Score: {result['score']:.4f}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Document: {metadata['document']}")
        print(f"Chunk: {metadata['chunk']}")

        print("\nText:")
        print(result["text"][:1000])


def main():

    retriever = Retriever()

    query = input("\nEnter your question: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = retriever.search(
        query=query,
        top_k=5,
    )

    print_results(results)


if __name__ == "__main__":
    main()