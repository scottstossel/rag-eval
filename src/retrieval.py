"""
Phase 2 — Retrieval Layer
=========================
  • Embed documents using sentence-transformers (all-MiniLM-L6-v2)
  • Normalize embeddings and build FAISS IndexFlatIP
  • Implement retrieve(query, k=5) -> list[dict]
  • Evaluate retrieval hit_rate@k, precision@k, and mrr@k
"""

import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 1. Retrieval system
# ---------------------------------------------------------------------------

class DenseRetriever:
    """
    Dense retrieval pipeline using sentence-transformers + FAISS IndexFlatIP.

    Usage
    -----
    retriever = DenseRetriever()
    retriever.fit_index(documents)      # list[{doc_id, text}]
    results = retriever.retrieve(query, k=5)
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"[retrieval] Loading model: {self.MODEL_NAME}")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.index = None          # faiss.IndexFlatIP
        self.doc_ids = []          # parallel list of doc_id strings
        self.documents = []        # original list[{doc_id, text}]

    def fit_index(self, documents: list[dict]) -> None:
        """Embed all documents, L2-normalise, and add to FAISS IndexFlatIP."""
        self.documents = documents
        texts = [d["text"] for d in documents]
        self.doc_ids = [d["doc_id"] for d in documents]

        print(f"[retrieval] Encoding {len(texts)} documents ...")
        t0 = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # L2-normalise so inner product == cosine similarity
        faiss.normalize_L2(embeddings)

        # Build IndexFlatIP
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        elapsed = time.perf_counter() - t0
        print(f"[retrieval] Index built: {self.index.ntotal} vectors "
              f"(dim={dim}) in {elapsed:.2f}s")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrieve top-k documents for a natural-language query.

        Returns
        -------
        list[dict] with keys: doc_id, score, text
        """
        q_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_vec)
        scores, indices = self.index.search(q_vec, k)   # shapes (1, k)

        doc_map = {d["doc_id"]: d["text"] for d in self.documents}
        return [
            {"doc_id": self.doc_ids[idx], "score": float(score), "text": doc_map[self.doc_ids[idx]]}
            for idx, score in zip(indices[0], scores[0])
            if idx != -1                               # FAISS returns -1 for padding
        ]


# ---------------------------------------------------------------------------
# 2. Evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(retriever: DenseRetriever, qa_pairs: list[dict], k: int = 5) -> dict:
    """
    Compute retrieval quality metrics over all QA pairs.

    Metrics
    -------
    hit_rate@k   fraction of queries where gold_doc_id is in the top-k
    precision@k  average fraction of top-k results that equal gold_doc_id
    mrr@k        mean reciprocal rank of the first correct hit
    """
    hits = 0
    precision_sum = 0.0
    reciprocal_ranks = []

    for qa in tqdm(qa_pairs, desc=f"Evaluating k={k}", leave=False):
        results = retriever.retrieve(qa["question"], k=k)
        retrieved_ids = [r["doc_id"] for r in results]
        gold = qa["gold_doc_id"]

        if gold in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(gold) + 1    # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        relevant = sum(1 for did in retrieved_ids if did == gold)
        precision_sum += relevant / k

    n = len(qa_pairs)
    return {
        "num_queries":    n,
        "k":              k,
        "hit_rate_at_k":  round(hits / n, 4),
        "precision_at_k": round(precision_sum / n, 4),
        "mrr_at_k":       round(float(np.mean(reciprocal_ranks)), 4),
        "hits":           hits,
    }


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    data_dir = Path("outputs")          # Phase 1 outputs
    out_dir  = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 outputs
    with open(data_dir / "documents.json") as f:
        documents = json.load(f)
    with open(data_dir / "qa_pairs.json") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(documents)} documents, {len(qa_pairs)} QA pairs.\n")

    # ---- Build index -------------------------------------------------------
    retriever = DenseRetriever()
    retriever.fit_index(documents)

    # ---- Sanity check — single query ---------------------------------------
    print("\n[demo] Sample retrieval:")
    sample_q    = qa_pairs[0]["question"]
    sample_gold = qa_pairs[0]["gold_doc_id"]
    results = retriever.retrieve(sample_q, k=5)
    print(f"  Question : {sample_q}")
    print(f"  Gold doc : {sample_gold}")
    for r in results:
        marker = "CORRECT" if r["doc_id"] == sample_gold else "      "
        print(f"  [{marker}] {r['doc_id']}  score={r['score']:.4f}  {r['text'][:70]}...")

    # ---- Evaluate at k = 1, 3, 5 -------------------------------------------
    print()
    all_results = {}
    for k in [1, 3, 5]:
        metrics = evaluate_retrieval(retriever, qa_pairs, k=k)
        all_results[f"k={k}"] = metrics
        print(f"  hit_rate@{k}   = {metrics['hit_rate_at_k']:.4f}  "
              f"({metrics['hits']}/{metrics['num_queries']})")
        print(f"  precision@{k}  = {metrics['precision_at_k']:.4f}")
        print(f"  mrr@{k}        = {metrics['mrr_at_k']:.4f}\n")

    # ---- Save outputs -------------------------------------------------------
    eval_path = out_dir / "retrieval_eval.json"
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[done] Evaluation results -> {eval_path}")

    retriever_path = out_dir / "retriever.pkl"
    with open(retriever_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"[done] Retriever serialised -> {retriever_path}")

    return retriever, all_results


if __name__ == "__main__":
    main()