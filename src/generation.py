"""
Phase 3 — Generation + Decomposed Metrics
==========================================
  • generate_answer(question, retrieved_contexts) via OpenAI
  • Grounding metric  : is the answer supported by the retrieved context?
  • Hallucination metric : is the answer contradicted by / absent from context?
  • Decompose every failure into one of three buckets:
      - retrieval_failure  : gold doc was not in top-k
      - grounding_failure  : gold doc was retrieved but answer isn't supported
      - generation_failure : answer present and grounded but still wrong
  • Saves outputs/generation_eval.json
"""

import json
from dotenv import load_dotenv
import os
import pickle
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
from src.retrieval import DenseRetriever

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL      = "gpt-4o-mini"          # cheap + capable; swap to gpt-4o if desired
K          = 5                       # retrieval depth (must match Phase 2)
SAMPLE_N   = 100                     # QA pairs to evaluate (full=1000, slow+costly)
SLEEP      = 0.1                     # seconds between API calls (rate-limit buffer)

load_dotenv()                       # load .env file if present
client = OpenAI()                    # reads OPENAI_API_KEY from environment


# ---------------------------------------------------------------------------
# 1. Generation
# ---------------------------------------------------------------------------

GENERATION_SYSTEM = """You are a precise question-answering assistant.
Answer the question using ONLY the provided context.
Be concise — answer in as few words as possible.
If the context does not contain the answer, reply exactly: "I don't know"."""

def generate_answer(question: str, retrieved_contexts: list[dict]) -> str:
    """
    Generate an answer given a question and list of retrieved context dicts.
    Each context dict has keys: doc_id, score, text.
    Returns the model's answer string.
    """
    context_block = "\n\n".join(
        f"[{i+1}] {ctx['text']}" for i, ctx in enumerate(retrieved_contexts)
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM},
            {"role": "user",   "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=64,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 2. Grounding judge
# ---------------------------------------------------------------------------

GROUNDING_SYSTEM = """You are an evaluation assistant.
Given a context, a question, and a generated answer, decide whether the answer
is supported by the context.

Reply with EXACTLY one word: "supported" or "unsupported".
- "supported"   : the answer can be directly verified from the context text
- "unsupported" : the answer contains information not found in, or contradicted by, the context"""

def is_grounded(question: str, context_text: str, answer: str) -> bool:
    """Returns True if the answer is supported by context_text."""
    if answer.lower() == "i don't know":
        return True     # abstention is always grounded
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": GROUNDING_SYSTEM},
            {"role": "user",   "content": (
                f"Context: {context_text}\n\n"
                f"Question: {question}\n"
                f"Answer: {answer}"
            )},
        ],
        temperature=0,
        max_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("supported")


# ---------------------------------------------------------------------------
# 3. Answer correctness check
# ---------------------------------------------------------------------------

def is_correct(generated: str, gold: str) -> bool:
    """
    Lexical correctness: gold answer appears in generated answer (case-insensitive).
    Simple but appropriate for SQuAD-style short answers.
    """
    return gold.lower().strip() in generated.lower().strip()


# ---------------------------------------------------------------------------
# 4. Failure decomposition
# ---------------------------------------------------------------------------

def classify_failure(
    gold_in_retrieved: bool,
    grounded: bool,
    correct: bool,
) -> str | None:
    """
    Decompose a wrong answer into one of three failure buckets.
    Returns None if the answer is correct.

    Priority order (mutually exclusive, hierarchical):
      1. retrieval_failure  — gold doc wasn't in top-k; generator had no chance
      2. grounding_failure  — gold doc retrieved but answer isn't grounded
      3. generation_failure — context was there and grounded but answer still wrong
    """
    if correct:
        return None
    if not gold_in_retrieved:
        return "retrieval_failure"
    if not grounded:
        return "grounding_failure"
    return "generation_failure"


# ---------------------------------------------------------------------------
# 5. Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(retriever, qa_pairs: list[dict], n: int = SAMPLE_N) -> dict:
    sample = qa_pairs[:n]
    results = []

    for qa in tqdm(sample, desc="Evaluating"):
        question   = qa["question"]
        gold_ans   = qa["answer"]
        gold_doc   = qa["gold_doc_id"]

        # -- Retrieve --------------------------------------------------------
        retrieved = retriever.retrieve(question, k=K)
        retrieved_ids = [r["doc_id"] for r in retrieved]
        gold_in_retrieved = gold_doc in retrieved_ids

        # -- Generate --------------------------------------------------------
        generated = generate_answer(question, retrieved)
        time.sleep(SLEEP)

        # -- Grounding -------------------------------------------------------
        # Judge against the concatenation of all retrieved contexts
        combined_context = " ".join(r["text"] for r in retrieved)
        grounded = is_grounded(question, combined_context, generated)
        time.sleep(SLEEP)

        # -- Correctness & failure decomposition -----------------------------
        correct = is_correct(generated, gold_ans)
        failure_type = classify_failure(gold_in_retrieved, grounded, correct)

        results.append({
            "question":           question,
            "gold_answer":        gold_ans,
            "gold_doc_id":        gold_doc,
            "generated_answer":   generated,
            "gold_in_retrieved":  gold_in_retrieved,
            "grounded":           grounded,
            "correct":            correct,
            "failure_type":       failure_type,   # None = correct
        })

    # -- Aggregate metrics ---------------------------------------------------
    n_total             = len(results)
    n_correct           = sum(1 for r in results if r["correct"])
    n_grounded          = sum(1 for r in results if r["grounded"])
    n_hallucinated      = sum(1 for r in results if not r["grounded"] and not r["correct"])
    n_retrieval_fail    = sum(1 for r in results if r["failure_type"] == "retrieval_failure")
    n_grounding_fail    = sum(1 for r in results if r["failure_type"] == "grounding_failure")
    n_generation_fail   = sum(1 for r in results if r["failure_type"] == "generation_failure")

    metrics = {
        "num_evaluated":        n_total,
        "k":                    K,
        "model":                MODEL,
        # Core metrics
        "accuracy":             round(n_correct / n_total, 4),
        "grounding_rate":       round(n_grounded / n_total, 4),
        "hallucination_rate":   round(n_hallucinated / n_total, 4),
        # Decomposed failures
        "failure_breakdown": {
            "retrieval_failure":  round(n_retrieval_fail / n_total, 4),
            "grounding_failure":  round(n_grounding_fail / n_total, 4),
            "generation_failure": round(n_generation_fail / n_total, 4),
        },
        "failure_counts": {
            "retrieval_failure":  n_retrieval_fail,
            "grounding_failure":  n_grounding_fail,
            "generation_failure": n_generation_fail,
            "correct":            n_correct,
        },
    }

    return metrics, results


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("outputs")
    out_dir  = Path("outputs")

    # Load Phase 1 + 2 outputs
    with open(data_dir / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    with open(data_dir / "retriever.pkl", "rb") as f:
        retriever = pickle.load(f)

    print(f"Loaded {len(qa_pairs)} QA pairs. Evaluating first {SAMPLE_N}.\n")

    metrics, results = evaluate(retriever, qa_pairs, n=SAMPLE_N)

    # -- Print summary -------------------------------------------------------
    print("\n=== Phase 3 Results ===")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Grounding rate    : {metrics['grounding_rate']:.4f}")
    print(f"  Hallucination rate: {metrics['hallucination_rate']:.4f}")
    print(f"\n  Failure breakdown ({metrics['num_evaluated']} questions):")
    for ftype, rate in metrics["failure_breakdown"].items():
        count = metrics["failure_counts"][ftype]
        print(f"    {ftype:<22}: {rate:.4f}  ({count})")
    print(f"    {'correct':<22}: {metrics['accuracy']:.4f}  ({metrics['failure_counts']['correct']})")

    # -- Save ----------------------------------------------------------------
    output = {"metrics": metrics, "results": results}
    eval_path = out_dir / "generation_eval.json"
    with open(eval_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] Full results -> {eval_path}")


if __name__ == "__main__":
    main()