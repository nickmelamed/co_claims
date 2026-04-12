import os
import sys
import json
import asyncio
import numpy as np
from tqdm import tqdm
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from gold.gold_evaluation import normalize_evidence
from eval.config import build_pipeline


# metric comparison
def metric_mae(pred, gold):
    errors = {}

    for k in gold:
        if k in pred:
            errors[k] = abs(pred[k] - gold[k])

    return errors


def aggregate_metric_errors(results):
    totals = {}
    counts = {}

    for r in results:
        errs = r["metric_errors"]

        for k, v in errs.items():
            totals[k] = totals.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1

    return {k: totals[k] / counts[k] for k in totals}


# main evaluation 
async def evaluate_end_to_end(dataset_path, retriever, output_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    pipeline = build_pipeline()
    results = []

    for row in tqdm(dataset):
        claim = row["claim"]

        # gold evidence
        gold = normalize_evidence(row["evidence"])

        # retrieved evidence
        retrieved = normalize_evidence(retriever.retrieve(claim))

        claim_time = datetime.now()

        # --- run pipeline ---
        oracle = await pipeline.run(claim, gold)
        rag = await pipeline.run(claim, retrieved)

        # metric comparison
        metric_errors = metric_mae(rag["metrics"], oracle["metrics"])

        # credibility gap
        gap = oracle["final_score"] - rag["final_score"]

        results.append({
            "claim": claim,
            "oracle_score": oracle["final_score"],
            "rag_score": rag["final_score"],
            "gap": gap,
            "metric_errors": metric_errors
        })

    # reporting
    avg_gap = np.mean([r["gap"] for r in results])
    metric_report = aggregate_metric_errors(results)

    print("\nEND-TO-END EVAL REPORT")
    print("=" * 40)
    print(f"Avg Credibility Gap: {avg_gap:.4f}")

    print("\nMetric MAE:")
    for k, v in sorted(metric_report.items(), key=lambda x: -x[1]):
        print(f"{k}: {v:.4f}")

    print("\nLARGE GAP CASES")
    print("=" * 40)

    for r in results:
        if r["gap"] > 0.2:
            print(f"- {r['claim']} (gap={r['gap']:.2f})")

    # save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {output_path}")