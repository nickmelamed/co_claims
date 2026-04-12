import os
import sys
import json
import hashlib
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from gold.gold_evaluation import normalize_evidence

# config
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.8
CACHE_PATH = "./gold/embedding_cache.json"
FAILURE_THRESHOLD = 0.5


# embedding cache 
class EmbeddingCache:
    def __init__(self, path=CACHE_PATH):
        self.path = Path(path)
        self.cache = self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.cache, f)

    def _key(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text):
        return self.cache.get(self._key(text))

    def set(self, text, embedding):
        self.cache[self._key(text)] = embedding


# embedder 
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.cache = EmbeddingCache()

    def embed(self, texts):
        embeddings = []

        for t in texts:
            cached = self.cache.get(t)
            if cached:
                embeddings.append(np.array(cached))
                continue

            emb = self.model.encode(t)
            self.cache.set(t, emb.tolist())
            embeddings.append(emb)

        return embeddings

    def save(self):
        self.cache.save()


# similarity
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


# retrieval metrics 
def compute_retrieval_metrics(gold, retrieved):
    matches = 0
    matched_retrieved = set()

    for g in gold:
        for j, r in enumerate(retrieved):
            sim = cosine(g["embedding"], r["embedding"])
            if sim >= SIM_THRESHOLD:
                matches += 1
                matched_retrieved.add(j)
                break

    recall = matches / max(1, len(gold))
    precision = len(matched_retrieved) / max(1, len(retrieved))

    return {
        "recall@k": recall,
        "precision@k": precision,
        "matches": matches,
        "gold_count": len(gold),
        "retrieved_count": len(retrieved),
    }


# reporting 
def aggregate_results(results):
    return {
        "avg_recall@k": np.mean([r["recall@k"] for r in results]),
        "avg_precision@k": np.mean([r["precision@k"] for r in results])
    }


def print_report(results):
    agg = aggregate_results(results)

    print("\nRETRIEVAL EVAL REPORT")
    print("=" * 40)
    print(f"Avg Recall@K:    {agg['avg_recall@k']:.4f}")
    print(f"Avg Precision@K: {agg['avg_precision@k']:.4f}")


def print_failures(results):
    print("\nLOW-RECALL CASES")
    print("=" * 40)

    for r in results:
        if r["recall@k"] < FAILURE_THRESHOLD:
            print(f"- {r['claim']} (recall={r['recall@k']:.2f})")


# main evaluation 
def evaluate_retrieval(dataset_path, retriever, output_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    embedder = Embedder()
    results = []

    for row in tqdm(dataset):
        claim = row["claim"]

        gold = normalize_evidence(row["evidence"])
        retrieved = normalize_evidence(retriever.retrieve(claim))

        gold_texts = [e["text"][:300] for e in gold]
        retrieved_texts = [e["text"][:300] for e in retrieved]

        gold_embs = embedder.embed(gold_texts)
        retrieved_embs = embedder.embed(retrieved_texts)

        for e, emb in zip(gold, gold_embs):
            e["embedding"] = emb

        for e, emb in zip(retrieved, retrieved_embs):
            e["embedding"] = emb

        metrics = compute_retrieval_metrics(gold, retrieved)

        results.append({
            "claim": claim,
            **metrics
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    embedder.save()

    print_report(results)
    print_failures(results)

    print(f"\nSaved → {output_path}")