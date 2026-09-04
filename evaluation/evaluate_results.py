import json
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "evaluation" / "baseline_results.json"
QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize(text):
    """Normalize text for simple evidence matching."""

    if text is None:
        return ""

    return " ".join(
        str(text)
        .lower()
        .replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .split()
    )


def retrieval_hit(retrieved_context, evidence):
    """
    Determine whether the retrieved context contains
    at least one of the expected evidence items.
    """

    if not evidence:
        return True

    combined_context = normalize(
        " ".join(
            chunk.get("page_content", "")
            for chunk in retrieved_context
        )
    )

    for item in evidence:
        if normalize(item) in combined_context:
            return True

    return False


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def evaluate():

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        questions = json.load(f)

    # Map question ID -> evaluation information
    question_map = {
        q["id"]: q
        for q in questions
    }

    retrieval_results = []

    for result in results:

        question_id = result["id"]

        question_info = question_map[question_id]

        retrieved_context = result.get(
            "retrieved_context",
            []
        )

        evidence = question_info.get(
            "evidence",
            []
        )

        hit = retrieval_hit(
            retrieved_context,
            evidence
        )

        retrieval_results.append({
            "id": question_id,
            "category": result["category"],
            "retrieval_hit": hit,
            "num_chunks": len(retrieved_context)
        })

    # -----------------------------------------------------
    # Overall retrieval metric
    # -----------------------------------------------------

    total = len(retrieval_results)

    hits = sum(
        r["retrieval_hit"]
        for r in retrieval_results
    )

    overall_hit_rate = hits / total if total else 0

    # -----------------------------------------------------
    # Category metrics
    # -----------------------------------------------------

    categories = {}

    for category in [
        "factual",
        "reasoning",
        "unanswerable",
        "adversarial"
    ]:

        category_results = [
            r
            for r in retrieval_results
            if r["category"] == category
        ]

        if not category_results:
            continue

        category_hits = sum(
            r["retrieval_hit"]
            for r in category_results
        )

        categories[category] = {
            "total": len(category_results),
            "hits": category_hits,
            "hit_rate": category_hits / len(category_results)
        }

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nOverall Retrieval Hit Rate")
    print("--------------------------")

    print(
        f"{hits}/{total} questions "
        f"= {overall_hit_rate * 100:.1f}%"
    )

    print("\nBy Category")
    print("--------------------------")

    for category, stats in categories.items():

        print(
            f"{category.capitalize():15} "
            f"{stats['hits']}/{stats['total']} "
            f"= {stats['hit_rate'] * 100:.1f}%"
        )

    print("\nPer-question Results")
    print("--------------------------")

    for r in retrieval_results:

        status = "HIT " if r["retrieval_hit"] else "MISS"

        print(
            f"{status} | "
            f"{r['id']:3} | "
            f"{r['category']:12} | "
            f"chunks={r['num_chunks']}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    evaluate()