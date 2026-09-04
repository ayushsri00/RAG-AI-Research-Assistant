import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "evaluation" / "baseline_results.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "answer_accuracy_results.json"

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from your environment."
    )


# Use the same model/provider as the RAG system.
JUDGE_MODEL = "openai/gpt-oss-120b"

judge_llm = ChatGroq(
    model=JUDGE_MODEL,
    temperature=0,
    #The reason is that we want the evaluator to behave as 
    # deterministically and consistently as possible.
    api_key=GROQ_API_KEY
)


# ---------------------------------------------------------
# LLM Judge
# ---------------------------------------------------------

def judge_answer(question, expected_answer, generated_answer):

    prompt = f"""
You are evaluating the answer produced by a RAG system.

Your job is to determine whether the GENERATED ANSWER correctly
answers the QUESTION according to the EXPECTED ANSWER.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

Evaluation rules:

1. Mark CORRECT if the generated answer contains the essential
   information required by the expected answer.

2. The generated answer does NOT need to use the exact same wording.

3. Additional information is acceptable if it does not contradict
   the expected answer.

4. For unanswerable questions, the expected answer may indicate
   that the information is unavailable. In that case, the generated
   answer should appropriately refuse to answer rather than invent
   unsupported information.

5. If the generated answer contradicts the expected answer,
   mark INCORRECT.

6. Do not use outside knowledge. Evaluate only against the
   QUESTION and EXPECTED ANSWER provided above.

Return ONLY valid JSON in this exact format:

{{
    "verdict": "CORRECT" or "INCORRECT",
    "reason": "short explanation"
}}
"""

    try:

        response = judge_llm.invoke(prompt)

        text = response.content.strip()

        # Remove markdown code fences if the model adds them.
        if text.startswith("```json"):
            text = text[7:]

        if text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        return json.loads(text)

    except Exception as e:

        return {
            "verdict": "ERROR",
            "reason": f"Judge failed: {str(e)}"
        }


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def run_evaluation():

    print("=" * 70)
    print("RAG ANSWER ACCURACY EVALUATION")
    print("=" * 70)

    # -----------------------------------------------------
    # Load baseline results
    # -----------------------------------------------------

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        results = json.load(f)

    print(f"\nQuestions: {len(results)}")
    print(f"Judge model: {JUDGE_MODEL}\n")

    evaluated_results = []

    # -----------------------------------------------------
    # Evaluate each answer
    # -----------------------------------------------------

    for i, item in enumerate(results, start=1):

        question_id = item["id"]
        category = item["category"]

        question = item["question"]
        expected_answer = item["expected_answer"]
        generated_answer = item.get("answer", "")

        print("-" * 70)
        print(f"[{i}/{len(results)}] {question_id}")
        print(f"Category: {category}")
        print(f"Question: {question}")

        if not generated_answer:

            verdict = "INCORRECT"
            reason = "No answer was generated."

        else:

            evaluation = judge_answer(
                question,
                expected_answer,
                generated_answer
            )

            verdict = evaluation.get(
                "verdict",
                "ERROR"
            )

            reason = evaluation.get(
                "reason",
                ""
            )

        item["answer_verdict"] = verdict
        item["answer_reason"] = reason

        evaluated_results.append(item)

        print(f"Verdict: {verdict}")
        print(f"Reason: {reason}")

    # -----------------------------------------------------
    # Calculate overall metrics
    # -----------------------------------------------------

    valid_results = [
        r
        for r in evaluated_results
        if r["answer_verdict"] in [
            "CORRECT",
            "INCORRECT"
        ]
    ]

    correct = sum(
        r["answer_verdict"] == "CORRECT"
        for r in valid_results
    )

    total = len(valid_results)

    overall_accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    # -----------------------------------------------------
    # Calculate category metrics
    # -----------------------------------------------------

    categories = [
        "factual",
        "reasoning",
        "unanswerable",
        "adversarial"
    ]

    category_metrics = {}

    for category in categories:

        category_results = [
            r
            for r in valid_results
            if r["category"] == category
        ]

        category_correct = sum(
            r["answer_verdict"] == "CORRECT"
            for r in category_results
        )

        category_total = len(category_results)

        category_accuracy = (
            category_correct / category_total * 100
            if category_total > 0
            else 0
        )

        category_metrics[category] = {
            "correct": category_correct,
            "total": category_total,
            "accuracy": round(
                category_accuracy,
                1
            )
        }

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("ANSWER ACCURACY RESULTS")
    print("=" * 70)

    print("\nOverall Answer Accuracy")
    print("-----------------------")

    print(
        f"{correct}/{total} = "
        f"{overall_accuracy:.1f}%"
    )

    print("\nBy Category")
    print("-----------------------")

    for category in categories:

        metrics = category_metrics[category]

        print(
            f"{category.capitalize():<15}"
            f"{metrics['correct']}/{metrics['total']} = "
            f"{metrics['accuracy']:.1f}%"
        )

    # -----------------------------------------------------
    # Per-question summary
    # -----------------------------------------------------

    print("\nPer-question Results")
    print("-----------------------")

    for result in evaluated_results:

        print(
            f"{result['answer_verdict']:<9} | "
            f"{result['id']:<3} | "
            f"{result['category']:<12}"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    output = {
        "overall": {
            "correct": correct,
            "total": total,
            "accuracy": round(
                overall_accuracy,
                1
            )
        },
        "by_category": category_metrics,
        "results": evaluated_results
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    run_evaluation()