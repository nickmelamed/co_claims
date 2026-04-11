"""
Gold Dataset Schema + Validation + Metric Evaluation Layer
"""

import json
from datetime import datetime

# schema definition 
REQUIRED_TOP_LEVEL = ["claim", "evidence"]

REQUIRED_EVIDENCE_FIELDS = [
    "text",
    "source_type"
]

OPTIONAL_EVIDENCE_FIELDS = [
    "domain",
    "timestamp",
    "relevance",
    "score"
]

EXPECTED_METRICS = [
    "ESS", "ECS", "EAS", "ERS", "ESTS",
    "EAGS", "SRS", "EVS", "CMS", "LCS", "HLS",
    "credibility"
]


# validation functions 
def validate_timestamp(ts):
    if ts is None:
        return True
    try:
        datetime.fromisoformat(ts)
        return True
    except:
        return False


def validate_evidence(evidence_list):
    errors = []

    if not isinstance(evidence_list, list) or len(evidence_list) == 0:
        return ["Evidence must be a non-empty list"]

    for i, e in enumerate(evidence_list):
        for field in REQUIRED_EVIDENCE_FIELDS:
            if field not in e:
                errors.append(f"Evidence[{i}] missing required field: {field}")

        if "timestamp" in e and not validate_timestamp(e.get("timestamp")):
            errors.append(f"Evidence[{i}] invalid timestamp format")

    return errors


def validate_expected(expected):
    errors = []

    if not expected:
        return errors  # optional for now

    for m in EXPECTED_METRICS:
        if m in expected:
            val = expected[m]
            if not (0 <= val <= 1):
                errors.append(f"Metric {m} must be between 0 and 1")

    return errors


def validate_row(row):
    errors = []

    # top-level
    for field in REQUIRED_TOP_LEVEL:
        if field not in row:
            errors.append(f"Missing required field: {field}")

    # evidence
    if "evidence" in row:
        errors.extend(validate_evidence(row["evidence"]))

    # claim_time
    if "claim_time" in row:
        if not validate_timestamp(row["claim_time"]):
            errors.append("Invalid claim_time format")

    # expected
    if "expected" in row:
        errors.extend(validate_expected(row["expected"]))

    return errors


def validate_dataset(dataset):
    all_errors = {}

    for i, row in enumerate(dataset):
        errors = validate_row(row)
        if errors:
            all_errors[i] = errors

    return all_errors


# enforcement/cleaning
def enforce_schema(row):
    """Auto-fix minor issues for robustness"""

    # claim_time fallback
    if "claim_time" not in row or not row["claim_time"]:
        row["claim_time"] = None

    # normalize evidence
    for e in row.get("evidence", []):
        if "domain" not in e:
            e["domain"] = "unknown"

        if "relevance" not in e:
            e["relevance"] = e.get("score", 0.5)

        if "timestamp" not in e:
            e["timestamp"] = None

    return row


def load_and_validate(path):
    with open(path, "r") as f:
        dataset = json.load(f)

    errors = validate_dataset(dataset)

    if errors:
        print("❌ Schema validation errors:")
        for idx, errs in errors.items():
            print(f"Row {idx}: {errs}")
    else:
        print("✅ Dataset schema valid")

    dataset = [enforce_schema(row) for row in dataset]

    return dataset


# metric evaluation
def metric_mae(pred, expected):
    """Mean Absolute Error per metric"""
    errors = {}

    for k, v in expected.items():
        if k not in pred:
            continue
        errors[k] = abs(pred[k] - v)

    return errors


def aggregate_metric_errors(results):
    """Aggregate MAE across dataset"""
    totals = {}
    counts = {}

    for r in results:
        pred = r.get("metrics", {})
        expected = r.get("expected", {})

        if not expected:
            continue

        errors = metric_mae(pred, expected)

        for k, v in errors.items():
            totals[k] = totals.get(k, 0) + v
            counts[k] = counts.get(k, 0) + 1

    return {
        k: totals[k] / counts[k]
        for k in totals
    }


def print_metric_report(results):
    report = aggregate_metric_errors(results)

    print("\n📊 METRIC EVALUATION REPORT")
    print("=" * 40)

    for k, v in sorted(report.items(), key=lambda x: -x[1]):
        print(f"{k}: MAE = {v:.4f}")


# integration 
def attach_expected(results, dataset):
    """Attach expected metrics back to results for evaluation"""

    for r, d in zip(results, dataset):
        if "expected" in d:
            r["expected"] = d["expected"]

    return results


