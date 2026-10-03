import json
from pathlib import Path

from app.clients.embedding_client import EmbeddingClient
from app.repositories.vector_repository import VectorRepository
from app.services.retrieval_service import RetrievalService


TOP_K = 3

DATASET_PATH = (
    Path(__file__).parent / "retrieval_dataset.json"
)


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # Initialize real retrieval components
    embedding_client = EmbeddingClient()
    vector_repository = VectorRepository()
    retrieval_service = RetrievalService(
        embedding_client=embedding_client,
        vector_repository=vector_repository,
    )

    # Load evaluation dataset
    dataset = load_dataset()

    hits = 0
    reciprocal_ranks = []

    print("\n=== Retrieval Evaluation ===\n")

    for index, item in enumerate(dataset, start=1):
        question = item["question"]

        expected_source = (
            item["expected_filename"],
            item["expected_page"],
        )

        # Run real retrieval
        results = retrieval_service.retrieve(
            question,
            top_k=TOP_K,
        )

        # Extract filename + page from retrieved chunks
        retrieved_sources = [
            (
                result.payload["filename"],
                result.payload["page"],
            )
            for result in results
        ]

        # -------------------------
        # Hit@K
        # -------------------------
        hit = expected_source in retrieved_sources

        if hit:
            hits += 1

        # -------------------------
        # Reciprocal Rank
        # -------------------------
        rank = None

        for rank_index, source in enumerate(
            retrieved_sources,
            start=1,
        ):
            if source == expected_source:
                rank = rank_index
                break

        reciprocal_rank = (
            1 / rank if rank is not None else 0
        )

        reciprocal_ranks.append(reciprocal_rank)

        # -------------------------
        # Per-question result
        # -------------------------
        print(f"Question {index}: {question}")
        print(f"Expected: {expected_source}")
        print(f"Retrieved: {retrieved_sources}")
        print(f"Hit@{TOP_K}: {hit}")
        print(f"Rank: {rank}")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")
        print("-" * 60)

    # -------------------------
    # Final metrics
    # -------------------------
    total_questions = len(dataset)

    hit_rate = (
        hits / total_questions
        if total_questions > 0
        else 0
    )

    mrr = (
        sum(reciprocal_ranks) / total_questions
        if total_questions > 0
        else 0
    )

    print("\n=== Summary ===")
    print(f"Questions: {total_questions}")
    print(f"Hits: {hits}")
    print(f"Hit@{TOP_K}: {hit_rate:.2%}")
    print(f"MRR: {mrr:.3f}")


if __name__ == "__main__":
    main()