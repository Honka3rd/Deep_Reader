from coordinator import Coordinator


def main() -> None:
    coordinator = Coordinator(
        chunk_size=300,
        chunk_overlap = 50
    )
    doc_name = "Madame Bovary"
    bundle = coordinator.get_bundle(doc_name)
    question: str = "Charles Bovary是個什麼樣的人？"
    response = bundle.answer(question)

    print("\n=== 回答 ===")
    print(response)

if __name__ == "__main__":
    main()