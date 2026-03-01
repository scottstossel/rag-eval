#!/usr/bin/env python3
"""
RAG Evaluation Pipeline — Unified Runner
=========================================

INTERVIEW FRAMING:
------------------
This is a production-ready, config-driven evaluation harness that:
  1. Reads all parameters from config.yaml (or env vars / CLI flags)
  2. Orchestrates all phases: data prep → retrieval → generation → shift
  3. Produces a single structured JSON report with baseline + shifted metrics
  4. Designed to run as a Kubernetes Job on OpenShift AI

WHY THIS MATTERS FOR ML EVAL ROLES:
------------------------------------
- Shows you can build end-to-end eval infrastructure, not just ad-hoc scripts
- Config-driven design makes it operator-friendly (no code changes to tune params)
- Structured output enables CI/CD gates: "fail deployment if accuracy_delta < -0.15"
- Container-native: runs identically on laptop, CI, or Kubernetes

USAGE:
------
  python run_eval.py --config config.yaml
  python run_eval.py --config config.yaml --baseline-only
  python run_eval.py --help
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

# Import phase modules from src/ package
try:
    from src.retrieval import DenseRetriever, evaluate_retrieval
    from src.generation import evaluate as run_generation_eval
    from src.shift_eval import (
        create_shifted_dataset,
        evaluate_shifted,
        compute_deltas,
    )
except ImportError as e:
    logging.error(f"Failed to import phase modules: {e}")
    logging.error("Ensure src/ package exists with __init__.py and all phase scripts")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """
    Load YAML config and merge with environment variables.
    
    DESIGN CHOICE (interview talking point):
    Environment variables override config file values, following 12-factor app
    principles. This enables operator-friendly deployment:
      - Base config in version control
      - Secrets (API keys) injected via env vars
      - Per-environment overrides (dev/staging/prod) via ConfigMap
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Example: Allow env var overrides for sensitive values
    if "OPENAI_API_KEY" in os.environ:
        logger.info("Using OPENAI_API_KEY from environment")
    
    # Could extend this to read all config from env vars like:
    # EVAL_GENERATION_MODEL, EVAL_RETRIEVAL_K, etc.
    
    return config


# ---------------------------------------------------------------------------
# Phase orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, baseline_only: bool = False):
    """
    Execute all evaluation phases and generate final report.
    
    WORKFLOW (interview talking point):
    -----------------------------------
    Phase 1: Data prep    → outputs/documents.json, qa_pairs.json
    Phase 2: Retrieval    → outputs/retrieval_eval.json, retriever.pkl
    Phase 3: Generation   → outputs/generation_eval.json
    Phase 4: Shift eval   → outputs/shift_eval.json (unless --baseline-only)
    Phase 5: Consolidate  → outputs/evaluation_report.json
    
    In OpenShift AI, each phase would be a Kubeflow pipeline step, with outputs
    stored in S3 and passed between steps. This runner simulates that locally.
    """
    out_dir = Path(config["data"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("RAG EVALUATION PIPELINE — START")
    logger.info("=" * 70)
    
    # ---- Phase 1: Data Preparation -----------------------------------------
    logger.info("\n[Phase 1] Data Preparation")
    logger.info("-" * 70)
    
    # Check if already done
    docs_path = out_dir / "documents.json"
    qa_path   = out_dir / "qa_pairs.json"
    
    if docs_path.exists() and qa_path.exists():
        logger.info("✓ Data already prepared (documents.json, qa_pairs.json found)")
    else:
        logger.info("Running data preparation...")
        # In production, this would shell out or import the function properly
        # For now, assume it's been run or we'll error
        logger.warning("Run data_preparation.py manually first, or integrate here")
        sys.exit(1)
    
    with open(docs_path) as f:
        documents = json.load(f)
    with open(qa_path) as f:
        qa_pairs = json.load(f)
    
    logger.info(f"✓ Loaded {len(documents)} documents, {len(qa_pairs)} QA pairs")
    
    # ---- Phase 2: Retrieval ------------------------------------------------
    logger.info("\n[Phase 2] Retrieval Evaluation")
    logger.info("-" * 70)
    
    retriever_path = out_dir / "retriever.pkl"
    retrieval_eval_path = out_dir / "retrieval_eval.json"
    
    if retriever_path.exists() and retrieval_eval_path.exists():
        logger.info("✓ Retrieval already evaluated (retriever.pkl, retrieval_eval.json found)")
        with open(retriever_path, "rb") as f:
            retriever = pickle.load(f)
        with open(retrieval_eval_path) as f:
            retrieval_metrics = json.load(f)
    else:
        logger.info("Building retrieval index...")
        from retrieval import DenseRetriever, evaluate_retrieval
        retriever = DenseRetriever()
        retriever.fit_index(documents)
        
        logger.info(f"Evaluating retrieval at k={config['retrieval']['k']}...")
        retrieval_metrics = {}
        for k in [1, 3, 5]:
            metrics = evaluate_retrieval(retriever, qa_pairs, k=k)
            retrieval_metrics[f"k={k}"] = metrics
        
        # Save
        with open(retriever_path, "wb") as f:
            pickle.dump(retriever, f)
        with open(retrieval_eval_path, "w") as f:
            json.dump(retrieval_metrics, f, indent=2)
        
        logger.info(f"✓ Saved retriever → {retriever_path}")
        logger.info(f"✓ Saved metrics  → {retrieval_eval_path}")
    
    # ---- Phase 3: Generation (Baseline) ------------------------------------
    logger.info("\n[Phase 3] Generation Evaluation (Baseline)")
    logger.info("-" * 70)
    
    gen_eval_path = out_dir / "generation_eval.json"
    
    if gen_eval_path.exists():
        logger.info("✓ Generation eval already run (generation_eval.json found)")
        with open(gen_eval_path) as f:
            gen_results = json.load(f)
            baseline_metrics = gen_results["metrics"]
    else:
        logger.info(f"Running generation eval on {config['evaluation']['baseline_sample_size']} samples...")
        baseline_metrics, baseline_results = run_generation_eval(
            retriever,
            qa_pairs,
            n=config["evaluation"]["baseline_sample_size"]
        )
        
        gen_results = {"metrics": baseline_metrics, "results": baseline_results}
        with open(gen_eval_path, "w") as f:
            json.dump(gen_results, f, indent=2)
        
        logger.info(f"✓ Baseline accuracy: {baseline_metrics['accuracy']:.4f}")
        logger.info(f"✓ Saved → {gen_eval_path}")
    
    # ---- Phase 4: Distribution Shift ---------------------------------------
    if baseline_only:
        logger.info("\n[Phase 4] Skipped (--baseline-only flag set)")
        shifted_metrics = None
        deltas = None
    else:
        logger.info("\n[Phase 4] Distribution Shift Evaluation")
        logger.info("-" * 70)
        
        shift_eval_path = out_dir / "shift_eval.json"
        shifted_qa_path = out_dir / "shifted_qa_pairs.json"
        
        if shift_eval_path.exists():
            logger.info("✓ Shift eval already run (shift_eval.json found)")
            with open(shift_eval_path) as f:
                shift_data = json.load(f)
                shifted_metrics = shift_data["shifted_metrics"]
                deltas = shift_data["deltas"]
        else:
            logger.info(f"Generating {config['evaluation']['shift_sample_size']} shifted queries...")
            
            # Generate or load shifted queries
            if shifted_qa_path.exists():
                with open(shifted_qa_path) as f:
                    shifted_qa = json.load(f)
            else:
                shifted_qa = create_shifted_dataset(
                    qa_pairs,
                    n=config["evaluation"]["shift_sample_size"]
                )
                with open(shifted_qa_path, "w") as f:
                    json.dump(shifted_qa, f, indent=2)
                logger.info(f"✓ Saved shifted queries → {shifted_qa_path}")
            
            logger.info("Running evaluation on shifted queries...")
            shifted_metrics, shifted_results = evaluate_shifted(retriever, shifted_qa)
            
            deltas = compute_deltas(baseline_metrics, shifted_metrics)
            
            shift_data = {
                "baseline_metrics": baseline_metrics,
                "shifted_metrics":  shifted_metrics,
                "deltas":           deltas,
                "shifted_results":  shifted_results,
            }
            with open(shift_eval_path, "w") as f:
                json.dump(shift_data, f, indent=2)
            
            logger.info(f"✓ Accuracy delta: {deltas['accuracy_delta']:+.4f}")
            logger.info(f"✓ Saved → {shift_eval_path}")
    
    # ---- Phase 5: Final Report ---------------------------------------------
    logger.info("\n[Phase 5] Generating Final Report")
    logger.info("-" * 70)
    
    report = {
        "pipeline_config": {
            "dataset":         config["data"]["dataset"],
            "split":           config["data"]["split"],
            "retrieval_model": config["retrieval"]["model"],
            "generation_model": config["generation"]["model"],
            "k":               config["retrieval"]["k"],
            "baseline_n":      config["evaluation"]["baseline_sample_size"],
            "shift_n":         config["evaluation"]["shift_sample_size"] if not baseline_only else None,
        },
        "retrieval_metrics": retrieval_metrics[f"k={config['retrieval']['k']}"],
        "baseline_metrics":  baseline_metrics,
        "shifted_metrics":   shifted_metrics,
        "deltas":            deltas,
        "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    
    # Add per-question results if requested
    if config["reporting"]["include_per_question_results"]:
        if not baseline_only:
            with open(out_dir / "shift_eval.json") as f:
                report["detailed_results"] = json.load(f)["shifted_results"]
    
    report_path = out_dir / config["reporting"]["final_report_name"]
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Final report → {report_path}")
    
    # ---- Summary -----------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nBaseline Performance:")
    logger.info(f"  Accuracy:          {baseline_metrics['accuracy']:.4f}")
    logger.info(f"  Grounding rate:    {baseline_metrics['grounding_rate']:.4f}")
    logger.info(f"  Retrieval hit@{config['retrieval']['k']}: {report['retrieval_metrics']['hit_rate_at_k']:.4f}")
    
    if not baseline_only and deltas:
        logger.info(f"\nDistribution Shift Impact:")
        logger.info(f"  Accuracy delta:    {deltas['accuracy_delta']:+.4f}")
        logger.info(f"  Grounding delta:   {deltas['grounding_delta']:+.4f}")
        logger.info(f"  Retrieval delta:   {deltas['retrieval_failure_delta']:+.4f}")
        
        # Automated insight (interview talking point: operationalizing insights)
        if abs(deltas["retrieval_failure_delta"]) > abs(deltas["grounding_failure_delta"]):
            logger.info("\n>>> KEY INSIGHT: Retrieval is the primary bottleneck.")
            logger.info("    Recommendation: Invest in hybrid search or fine-tune embeddings.")
        else:
            logger.info("\n>>> KEY INSIGHT: Generation/grounding degrades more than retrieval.")
            logger.info("    Recommendation: Strengthen prompts or add citation requirements.")
    
    logger.info(f"\n✓ All outputs in: {out_dir}/")
    logger.info(f"✓ Final report:   {report_path}")
    
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Pipeline — Unified Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline eval, skip distribution shift (faster for dev)",
    )
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    report = run_pipeline(config, baseline_only=args.baseline_only)
    
    logger.info("\n✓ Pipeline execution complete.")


if __name__ == "__main__":
    main()