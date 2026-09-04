import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------
# Allow importing app.py from the project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import get_retrieval_chain
from questions import EVAL_QUESTIONS #using the questions.py file.


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CORPUS_PATH = PROJECT_ROOT / "Text Corpus examples" / "Corpus.pdf"

MODEL_NAME = "openai/gpt-oss-120b"

# IMPORTANT:
# This must be different from your normal application index.
EVAL_INDEX_NAME = "research-rag-eval" 
# Because we don't want evaluation data contaminating the production/application index.
# This gives us isolation between the application and evaluation environments.

OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "baseline_results.json"


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def run_evaluation():

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found at: {CORPUS_PATH}"
        )

    print("=" * 70)
    print("RAG BASELINE EVALUATION")
    print("=" * 70)

    print(f"\nCorpus: {CORPUS_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Evaluation index: {EVAL_INDEX_NAME}")
    print(f"Questions: {len(EVAL_QUESTIONS)}")

    print("\nBuilding evaluation RAG pipeline...")

    # Open the corpus as a file-like object.
    with open(CORPUS_PATH, "rb") as uploaded_file:

        # Create the same RAG pipeline used by the application,
        # but point it at a dedicated evaluation index.
        chain = get_retrieval_chain(
            uploaded_file,
            MODEL_NAME,
            index_name=EVAL_INDEX_NAME
        )

        print("RAG pipeline ready.\n")

        results = []

        for i, item in enumerate(EVAL_QUESTIONS, start=1):

            question_id = item["id"]
            category = item["category"]
            question = item["question"]
            expected_answer = item["expected_answer"]

            print("-" * 70)
            print(f"[{i}/{len(EVAL_QUESTIONS)}] {question_id}")
            print(f"Category: {category}")
            print(f"Question: {question}")

            try:

                result = chain.invoke({
                    "input": question,
                    "additional_context": ""
                })

                answer = result.get("answer", "")

                # create_retrieval_chain returns the retrieved
                # documents under "context".
                retrieved_docs = result.get("context", [])
                #his is extremely useful because we can now evaluate retrieval independently from answer generation.

                retrieved_context = []

                for doc in retrieved_docs:

                    retrieved_context.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    })

                evaluation_result = {
                    "id": question_id,
                    "category": category,
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": answer,
                    "retrieved_context": retrieved_context
                }

                results.append(evaluation_result)

                print(f"Answer: {answer}")
                print(
                    f"Retrieved chunks: {len(retrieved_context)}"
                )

            except Exception as e:

                print(f"ERROR: {e}")

                results.append({
                    "id": question_id,
                    "category": category,
                    "question": question,
                    "expected_answer": expected_answer,
                    "answer": None,
                    "retrieved_context": [],
                    "error": str(e)
                })

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(OUTPUT_PATH)

    print("\nWe have NOT calculated metrics yet.")
    print("Next step: inspect retrieval + answers and add automated evaluation.")


if __name__ == "__main__":
    run_evaluation()