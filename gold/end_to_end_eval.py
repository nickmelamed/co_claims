import os
import sys
import json
import asyncio
import numpy as np
from tqdm import tqdm
from datetime import datetime
import traceback

from dotenv import load_dotenv
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from RAGService import retrieve_fn
from eval.config import build_pipeline
from eval.evaluator.deterministic.source_types import extract_domain

# evidence normalization
def parse_time_safe(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except:
        return None

def normalize_evidence(evidence_list):
    normalized = []

    for e in evidence_list:
        url = e.get("url", "")
        raw_type = (e.get("source_type") or "").lower()

        domain = e.get("domain")
        if not domain or domain == "unknown":
            if url:
                domain = extract_domain(url)
            else:
                domain = "unknown"

        # source type normalization
        if raw_type in ["10-k", "10k", "10-q", "10q"]:
            source_type = "financial_filing"
        elif raw_type in ["news", "news_article"]:
            source_type = "news_article"
        else:
            source_type = raw_type or "unknown"

        relevance = e.get("relevance")
        if relevance is None:
            relevance = e.get("score", 0.5)

        normalized.append({
            "text": e.get("text", ""),
            "timestamp": parse_time_safe(e.get("timestamp") or e.get("date")),
            "domain": domain,
            "source_type": source_type,
            "relevance": relevance
        })

    return normalized


# metric comparison
def metric_mae(pred, gold):
    errors = {}

    for k in gold:
        if k in pred:
            errors[k] = abs(pred[k] - gold[k])

    return errors

def mean_absolute_error(metric_errors):
    if not metric_errors:
        return 0.0
    return np.mean(list(metric_errors.values()))

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
async def evaluate_end_to_end(dataset_path, retrieve_fn, output_path):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # build pipeline 
    baseline_pipeline = build_pipeline("baseline", retrieve_fn)
    single_pipeline = build_pipeline("single_llm", retrieve_fn)
    full_pipeline = build_pipeline("full", retrieve_fn)

    # parallel retrieval
    #print("\n🔍 Running parallel retrieval...")

    retrieval_tasks = [
        retrieve_fn(row["claim"])
        for row in dataset
    ]

    retrieval_results = await asyncio.gather(*retrieval_tasks)

    # bounded parallel pipeline 
    SEM_LIMIT = 5
    semaphore = asyncio.Semaphore(SEM_LIMIT)

    async def process_row(row, retrieved_raw):
        async with semaphore:

            claim = row["claim"]

            gold = normalize_evidence(row["evidence"])

            if not retrieved_raw:
                print(f" No retrieval results for: {claim}")
                retrieved_raw = []

            retrieved = normalize_evidence(retrieved_raw)

            # Oracle
            oracle_full = await full_pipeline.run(claim, gold)

            # RAG
            baseline = await baseline_pipeline.run(claim, retrieved)
            single = await single_pipeline.run(claim, retrieved)
            full = await full_pipeline.run(claim, retrieved)

            # metric comparison
            baseline_metric_errors = metric_mae(baseline["metrics"], oracle_full["metrics"])
            single_metric_errors = metric_mae(single["metrics"], oracle_full["metrics"])
            full_metric_errors = metric_mae(full["metrics"], oracle_full["metrics"])

            # credibility gap
            baseline_gap = oracle_full["credibility"] - baseline["credibility"]
            single_gap = oracle_full["credibility"] - single["credibility"]
            full_gap = oracle_full["credibility"] - full["credibility"]

            # average MAE 
            baseline_mae = mean_absolute_error(baseline_metric_errors)
            single_mae = mean_absolute_error(single_metric_errors)
            full_mae = mean_absolute_error(full_metric_errors)

            return {
                "claim": claim,
                "oracle_score": oracle_full["credibility"],
                "baseline_score": baseline["credibility"],
                "single_llm_score": single["credibility"],
                "full_score": full["credibility"],
                "baseline_metric_errors": baseline_metric_errors,
                "single_metric_errors": single_metric_errors,
                "full_metric_errors": full_metric_errors,
                "baseline_mae": baseline_mae,
                "single_mae": single_mae,
                "full_mae": full_mae,
                "baseline_gap": baseline_gap,
                "single_gap": single_gap,
                "full_gap": full_gap,
            }

    # run tasks 
    print("\n⚙️ Running pipeline (parallel with limits)...")

    tasks = [
        process_row(row, retrieved_raw)
        for row, retrieved_raw in zip(dataset, retrieval_results)
    ]

    results = []

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        results.append(result)

    # reporting
    print("\n📊 END-TO-END TIER REPORT")
    print("=" * 50)

    avg_baseline = np.mean([r["baseline_gap"] for r in results])
    avg_single = np.mean([r["single_gap"] for r in results])
    avg_full = np.mean([r["full_gap"] for r in results])

    print("\nCredibility Gap (vs Oracle):")
    print(f"Baseline:    {avg_baseline:.4f}")
    print(f"Single LLM:  {avg_single:.4f}")
    print(f"Full System: {avg_full:.4f}")

    avg_baseline_mae = np.mean([r["baseline_mae"] for r in results])
    avg_single_mae = np.mean([r["single_mae"] for r in results])
    avg_full_mae = np.mean([r["full_mae"] for r in results])

    print("\n📏 Mean Absolute Error (vs Oracle Metrics):")
    print("=" * 50)
    print(f"Baseline MAE:    {avg_baseline_mae:.4f}")
    print(f"Single LLM MAE:  {avg_single_mae:.4f}")
    print(f"Full System MAE: {avg_full_mae:.4f}")

    # improvement breakdown 
    llm_improvements = []
    escalation_improvements = []

    for r in results:
        llm_improvements.append(r["baseline_gap"] - r["single_gap"])
        escalation_improvements.append(r["single_gap"] - r["full_gap"])

    print("\n Improvement Breakdown:")
    print("=" * 50)
    print(f"Avg LLM Gain:        {np.mean(llm_improvements):.4f}")
    print(f"Avg Escalation Gain: {np.mean(escalation_improvements):.4f}")

    # failure analysis
    print("\n FAILURE CASES")
    print("=" * 50)

    for r in results:
        if r["full_gap"] > 0.2:
            print(f"[FULL FAIL] {r['claim']} (gap={r['full_gap']:.2f})")

    # value diagnostics
    llm_help_count = sum(
        1 for r in results if (r["baseline_gap"] - r["single_gap"]) > 0.1
    )

    esc_help_count = sum(
        1 for r in results if (r["single_gap"] - r["full_gap"]) > 0.1
    )

    n = len(results)

    print("\n WHERE DOES VALUE COME FROM?")
    print("=" * 50)
    print(f"LLM helps in {llm_help_count}/{n} cases ({llm_help_count/n:.1%})")
    print(f"Escalation helps in {esc_help_count}/{n} cases ({esc_help_count/n:.1%})")

    # save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {output_path}")

# entry

if __name__ == "__main__":
    
    try:
        asyncio.run(
            evaluate_end_to_end(
                dataset_path="./gold/gold_dataset.json",
                retrieve_fn=retrieve_fn,
                output_path="./gold/end_to_end_results.json"
            )
        )
    except Exception as e:
        print("ERROR:")
        traceback.print_exc()
    