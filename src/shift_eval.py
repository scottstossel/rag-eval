"""
Phase 4 — Distribution Shift Evaluation
========================================

PURPOSE:
----------------------------
Test RAG robustness when queries don't match training distribution.
In production, users paraphrase questions, use synonyms, vary formality.

This phase answers: "Which component (retrieval, grounding, or generation)
degrades most under distribution shift?"

APPROACH:
---------
1. Generate ~150 shifted queries via LLM-based paraphrasing
   - Semantic equivalence (same answer)
   - Lexical/syntactic divergence (different wording)
   - Trade-off: expensive but high-quality vs cheap rule-based methods

2. Re-run full pipeline on shifted queries
3. Compute deltas: retrieval_delta, grounding_delta, accuracy_delta
4. Identify the brittleness bottleneck


"""

import json
import os
import pickle
import random
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm
# Import Phase 3 evaluation machinery
from src.retrieval import DenseRetriever
from src.generation import (
    generate_answer,
    is_grounded,
    is_correct,
    classify_failure,
    MODEL,
    K,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SHIFTED = 150          # Number of shifted queries to generate
SLEEP     = 0.1          # Rate limit buffer

client = OpenAI()

# ---------------------------------------------------------------------------
# 1. Query shift generation
# ---------------------------------------------------------------------------

# DESIGN CHOICE (interview talking point):
# We use LLM-based paraphrasing because:
#   - Preserves semantic meaning (same answer)
#   - Creates lexical divergence (different surface form)
#   - Controllable via prompt (formal/casual, verbose/terse)
# Alternative: rule-based (synonym replacement) is cheaper but brittle

SHIFT_SYSTEM = """You are a query paraphrasing assistant.
Given a question, generate a paraphrased version that:
1. Asks for the SAME information (semantic equivalence)
2. Uses different wording, sentence structure, or phrasing (lexical divergence)
3. Maintains natural English

Generate ONLY the paraphrased question, nothing else."""

def generate_shifted_query(original_question: str) -> str:
    """
    Generate a single shifted version of the question.
    
    Why LLM-based paraphrasing?
    - High semantic fidelity (preserves answer)
    - Natural lexical variation (tests retrieval robustness)
    - Controllable via prompt engineering
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SHIFT_SYSTEM},
            {"role": "user",   "content": original_question},
        ],
        temperature=0.7,     # Higher temp → more variation
        max_tokens=64,
    )
    return response.choices[0].message.content.strip()


def create_shifted_dataset(qa_pairs: list[dict], n: int = N_SHIFTED) -> list[dict]:
    """
    Sample n questions and generate shifted versions.
    Returns list of {question_original, question_shifted, answer, gold_doc_id}.
    
    DESIGN CHOICE: We sample randomly rather than taking the first n
    to avoid bias toward any particular topic distribution in SQuAD.
    """
    sample = random.sample(qa_pairs, min(n, len(qa_pairs)))
    shifted = []
    
    for qa in tqdm(sample, desc="Generating shifted queries"):
        shifted_q = generate_shifted_query(qa["question"])
        time.sleep(SLEEP)
        shifted.append({
            "question_original": qa["question"],
            "question_shifted":  shifted_q,
            "answer":            qa["answer"],
            "gold_doc_id":       qa["gold_doc_id"],
        })
    
    return shifted


# ---------------------------------------------------------------------------
# 2. Evaluation on shifted queries
# ---------------------------------------------------------------------------

def evaluate_shifted(retriever, shifted_qa: list[dict]) -> tuple[dict, list[dict]]:
    """
    Run full RAG pipeline on shifted queries.
    Returns (metrics, per_question_results).
    
    CRITICAL: We use the SAME retriever, generator, and evaluation logic
    as Phase 3. Only the queries change. This isolates distribution shift
    as the independent variable.
    """
    results = []
    
    for qa in tqdm(shifted_qa, desc="Evaluating shifted queries"):
        question   = qa["question_shifted"]   # USE SHIFTED VERSION
        gold_ans   = qa["answer"]
        gold_doc   = qa["gold_doc_id"]
        
        # -- Retrieve (same as Phase 3) --------------------------------------
        retrieved = retriever.retrieve(question, k=K)
        retrieved_ids = [r["doc_id"] for r in retrieved]
        gold_in_retrieved = gold_doc in retrieved_ids
        
        # -- Generate --------------------------------------------------------
        generated = generate_answer(question, retrieved)
        time.sleep(SLEEP)
        
        # -- Grounding -------------------------------------------------------
        combined_context = " ".join(r["text"] for r in retrieved)
        grounded = is_grounded(question, combined_context, generated)
        time.sleep(SLEEP)
        
        # -- Correctness & failure decomposition -----------------------------
        correct = is_correct(generated, gold_ans)
        failure_type = classify_failure(gold_in_retrieved, grounded, correct)
        
        results.append({
            "question_original":  qa["question_original"],
            "question_shifted":   question,
            "gold_answer":        gold_ans,
            "gold_doc_id":        gold_doc,
            "generated_answer":   generated,
            "gold_in_retrieved":  gold_in_retrieved,
            "grounded":           grounded,
            "correct":            correct,
            "failure_type":       failure_type,
        })
    
    # -- Aggregate metrics ---------------------------------------------------
    n_total            = len(results)
    n_correct          = sum(1 for r in results if r["correct"])
    n_grounded         = sum(1 for r in results if r["grounded"])
    n_hallucinated     = sum(1 for r in results if not r["grounded"] and not r["correct"])
    n_retrieval_fail   = sum(1 for r in results if r["failure_type"] == "retrieval_failure")
    n_grounding_fail   = sum(1 for r in results if r["failure_type"] == "grounding_failure")
    n_generation_fail  = sum(1 for r in results if r["failure_type"] == "generation_failure")
    
    metrics = {
        "num_evaluated":      n_total,
        "k":                  K,
        "model":              MODEL,
        "accuracy":           round(n_correct / n_total, 4),
        "grounding_rate":     round(n_grounded / n_total, 4),
        "hallucination_rate": round(n_hallucinated / n_total, 4),
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
# 3. Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(baseline_metrics: dict, shifted_metrics: dict) -> dict:
    """
    Compute degradation from baseline to shifted distribution.
    
    INTERVIEW TALKING POINT:
    Negative deltas indicate degradation (worse performance on shifted queries).
    The magnitude tells you which component is most brittle:
      - Large retrieval_delta → embeddings are brittle to paraphrasing
      - Large grounding_delta → generator hallucinations increase under shift
      - Large generation_delta → model struggles to extract from context
    
    This decomposition guides where to invest: better embeddings vs better prompts.
    """
    return {
        "accuracy_delta": round(
            shifted_metrics["accuracy"] - baseline_metrics["accuracy"], 4
        ),
        "grounding_delta": round(
            shifted_metrics["grounding_rate"] - baseline_metrics["grounding_rate"], 4
        ),
        "hallucination_delta": round(
            shifted_metrics["hallucination_rate"] - baseline_metrics["hallucination_rate"], 4
        ),
        "retrieval_failure_delta": round(
            shifted_metrics["failure_breakdown"]["retrieval_failure"]
            - baseline_metrics["failure_breakdown"]["retrieval_failure"],
            4,
        ),
        "grounding_failure_delta": round(
            shifted_metrics["failure_breakdown"]["grounding_failure"]
            - baseline_metrics["failure_breakdown"]["grounding_failure"],
            4,
        ),
        "generation_failure_delta": round(
            shifted_metrics["failure_breakdown"]["generation_failure"]
            - baseline_metrics["failure_breakdown"]["generation_failure"],
            4,
        ),
    }


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("outputs")
    out_dir  = Path("outputs")
    
    # Load Phase 1, 2, 3 artifacts
    with open(data_dir / "qa_pairs.json") as f:
        qa_pairs = json.load(f)
    
    with open(data_dir / "retriever.pkl", "rb") as f:
        retriever = pickle.load(f)
    
    with open(data_dir / "generation_eval.json") as f:
        baseline_results = json.load(f)
        baseline_metrics = baseline_results["metrics"]
    
    print(f"Loaded {len(qa_pairs)} QA pairs.")
    print(f"Baseline accuracy: {baseline_metrics['accuracy']:.4f}\n")
    
    # -- Step 1: Generate shifted queries ------------------------------------
    print(f"Generating {N_SHIFTED} shifted queries...")
    shifted_qa = create_shifted_dataset(qa_pairs, n=N_SHIFTED)
    
    shifted_qa_path = out_dir / "shifted_qa_pairs.json"
    with open(shifted_qa_path, "w") as f:
        json.dump(shifted_qa, f, indent=2)
    print(f"Saved shifted queries -> {shifted_qa_path}\n")
    
    # -- Step 2: Evaluate on shifted queries ---------------------------------
    print("Evaluating on shifted queries...")
    shifted_metrics, shifted_results = evaluate_shifted(retriever, shifted_qa)
    
    # -- Step 3: Compute deltas ----------------------------------------------
    deltas = compute_deltas(baseline_metrics, shifted_metrics)
    
    # -- Print results -------------------------------------------------------
    print("\n=== Phase 4 Results ===")
    print(f"\nBaseline (in-distribution):")
    print(f"  Accuracy          : {baseline_metrics['accuracy']:.4f}")
    print(f"  Grounding rate    : {baseline_metrics['grounding_rate']:.4f}")
    print(f"  Retrieval failure : {baseline_metrics['failure_breakdown']['retrieval_failure']:.4f}")
    
    print(f"\nShifted (out-of-distribution):")
    print(f"  Accuracy          : {shifted_metrics['accuracy']:.4f}")
    print(f"  Grounding rate    : {shifted_metrics['grounding_rate']:.4f}")
    print(f"  Retrieval failure : {shifted_metrics['failure_breakdown']['retrieval_failure']:.4f}")
    
    print(f"\nDegradation (Δ):")
    print(f"  Accuracy          : {deltas['accuracy_delta']:+.4f}")
    print(f"  Grounding         : {deltas['grounding_delta']:+.4f}")
    print(f"  Hallucination     : {deltas['hallucination_delta']:+.4f}")
    print(f"  Retrieval failure : {deltas['retrieval_failure_delta']:+.4f}")
    print(f"  Grounding failure : {deltas['grounding_failure_delta']:+.4f}")
    print(f"  Generation failure: {deltas['generation_failure_delta']:+.4f}")
    
    # INTERVIEW INSIGHT:
    if abs(deltas["retrieval_failure_delta"]) > abs(deltas["grounding_failure_delta"]):
        print("\n>>> KEY INSIGHT: Retrieval is the primary bottleneck under distribution shift.")
        print("    Recommendation: Invest in hybrid search (BM25 + dense) or fine-tune embeddings.")
    else:
        print("\n>>> KEY INSIGHT: Grounding/generation degrades more than retrieval.")
        print("    Recommendation: Strengthen generator prompts or add citation requirements.")
    
    # -- Save ----------------------------------------------------------------
    output = {
        "baseline_metrics": baseline_metrics,
        "shifted_metrics":  shifted_metrics,
        "deltas":           deltas,
        "shifted_results":  shifted_results,
    }
    
    eval_path = out_dir / "shift_eval.json"
    with open(eval_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] Full shift evaluation -> {eval_path}")


if __name__ == "__main__":
    random.seed(42)      # Reproducibility
    main()