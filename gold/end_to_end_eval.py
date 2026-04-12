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

    baseline_pipeline = build_pipeline("baseline")
    single_pipeline = build_pipeline("single_llm")
    full_pipeline = build_pipeline("full")

    results = []

    for row in tqdm(dataset):
        claim = row["claim"]

        # gold evidence
        gold = normalize_evidence(row["evidence"])

        # retrieved evidence
        retrieved = normalize_evidence(retriever.retrieve(claim))

        claim_time = datetime.now()

        # run all pipelinees 
        # oracle (gold evidence)
        oracle_full = await full_pipeline.run(claim, gold)

        # RAG (retrieved evidence)
        baseline = await baseline_pipeline.run(claim, retrieved)
        single = await single_pipeline.run(claim, retrieved)
        full = await full_pipeline.run(claim, retrieved)

        # metric comparison
        baseline_metric_errors = metric_mae(baseline["metrics"], oracle_full["metrics"])
        single_metric_errors = metric_mae(single['metrics'], oracle_full['metrics'])
        full_metric_errors = metric_mae(full['metrics'], oracle_full['metrics'])

        # credibility gap
        baseline_gap = oracle_full["credibility"] - baseline["credibility"]
        single_gap = oracle_full["credibility"] - single["credibility"]
        full_gap = oracle_full["credibility"] - full["credibility"]

        results.append({
    "claim": claim,

    "oracle_score": oracle_full["credibility"],

    "baseline_score": baseline["credibility"],
    "single_llm_score": single["credibility"],
    "full_score": full["credibility"],

    "baseline_metric_error": baseline_metric_errors,
    "single_metric_error": single_metric_errors,
    "full_metric_error": full_metric_errors,

    "baseline_gap": baseline_gap,
    "single_gap": single_gap,
    "full_gap": full_gap,
})

    # reporting

    print("\nEND-TO-END TIER REPORT")
    print("=" * 50)

    # credibility gaps
    avg_baseline = np.mean([r["baseline_gap"] for r in results])
    avg_single = np.mean([r["single_gap"] for r in results])
    avg_full = np.mean([r["full_gap"] for r in results])

    print("\nCredibility Gap (vs Oracle):")
    print(f"Baseline:    {avg_baseline:.4f}")
    print(f"Single LLM:  {avg_single:.4f}")
    print(f"Full System: {avg_full:.4f}")

    # improvement breakdown 
    llm_improvements = []
    escalation_improvements = []

    for r in results:
        llm_gain = r["baseline_gap"] - r["single_gap"]
        esc_gain = r["single_gap"] - r["full_gap"]

        llm_improvements.append(llm_gain)
        escalation_improvements.append(esc_gain)

    print("\n📈 Improvement Breakdown:")
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
    print("\n🔍 WHERE DOES VALUE COME FROM?")
    print("=" * 50)

    llm_help_count = 0
    esc_help_count = 0

    for r in results:
        if r["baseline_gap"] - r["single_gap"] > 0.1:
            llm_help_count += 1

        if r["single_gap"] - r["full_gap"] > 0.1:
            esc_help_count += 1

    n = len(results)

    print(f"LLM helps in {llm_help_count}/{n} cases ({llm_help_count/n:.1%})")
    print(f"Escalation helps in {esc_help_count}/{n} cases ({esc_help_count/n:.1%})")

    # Metric MAE 
    if "metric_errors_full" in results[0]:
        def aggregate_metric_errors_key(key):
            totals = {}
            counts = {}

            for r in results:
                errs = r.get(key, {})
                for k, v in errs.items():
                    totals[k] = totals.get(k, 0) + v
                    counts[k] = counts.get(k, 0) + 1

            return {k: totals[k] / counts[k] for k in totals}

        print("\nFULL SYSTEM METRIC MAE:")
        print("=" * 50)

        metric_report = aggregate_metric_errors_key("metric_errors_full")

        for k, v in sorted(metric_report.items(), key=lambda x: -x[1]):
            print(f"{k}: {v:.4f}")

    # save 
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved → {output_path}")