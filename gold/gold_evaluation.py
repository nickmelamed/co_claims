import cohere
import json
import os
from datetime import datetime

from eval.evaluator.llm.metrics import UNIFIED_PROMPT
from eval.evaluator.deterministic.metrics import DeterministicMetrics
from eval.evaluator.executor import UnifiedExecutor

from eval.evaluator.aggregator import Aggregator
from eval.evaluator.deterministic.source_types import get_type_weight, is_verifiable, extract_domain
from eval.structuring.entity_resolver import EntityResolver



class CohereJudge:
    def __init__(self, api_key, model="command-a-03-2025"):
        self.client = cohere.Client(api_key)
        self.model = model

    def evaluate(self, claim, evidence_list, relevances):
        metrics = ["ESS", "ECS", "CMS", "LCS", "HLS"]

        final_scores = {m: 0.0 for m in metrics}
        final_variances = {m: 0.0 for m in metrics}

        weight_sum = sum(relevances) + 1e-6
        if weight_sum <= 1e-6:
            weight_sum = len(evidence_list) or 1
            relevances = [1.0] * len(evidence_list)

        raw_outputs = []

        for e, r in zip(evidence_list, relevances):
            prompt = UNIFIED_PROMPT.format(
                claim=claim,
                evidence=e["text"][:1000]  # truncate
            )

            response = self.client.chat(
                model=self.model,
                message=prompt,
                temperature=0.0
            )

            parsed = self._parse(response.text)
            if not parsed:
                continue

            raw_outputs.append(parsed)

            for m in metrics:
                score = parsed.get(m, {}).get("score", 0.0)
                confidence = parsed.get(m, {}).get("confidence", 0.0)

                final_scores[m] += score * r
                final_variances[m] += (1 - confidence)

        # normalize
        for m in metrics:
            final_scores[m] /= weight_sum
            final_variances[m] /= max(1, len(evidence_list))

        return final_scores, final_variances, raw_outputs

    def _parse(self, text):
        try:
            if "<json>" in text:
                text = text.split("<json>")[-1]

            return json.loads(text.strip())
        except:
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group())
        return None



def parse_time_safe(date_str):
    """Convert date → datetime safely"""
    if not date_str or date_str != date_str:
        return datetime.now()
    try:
        return datetime.fromisoformat(date_str)
    except:
        return datetime.now()


def normalize_evidence(evidence_list, claim_time):
    """Ensure all required fields exist for executor"""
    normalized = []

    for e in evidence_list:
        normalized.append({
            "text": e.get("text", ""),
            "timestamp": parse_time_safe(e.get("date")),
            "domain": extract_domain(e.get("url")),
            "source_type": e.get("source_type", "unknown"),
            "relevance": e.get("relevance", 0.5),
            "support_score": e.get("support_score", 0.5)
        })

    return normalized


# main evaluation 

def evaluate_dataset(
    dataset_path,
    output_path,
    resolver,
    aggregator,
    type_weight_fn,
    verifiable_fn
):
    # load dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # init components
    det = DeterministicMetrics(
        type_weight_fn=type_weight_fn,
        verifiable_fn=verifiable_fn
    )

    llm_judge = CohereJudge(os.getenv("COHERE_KEY"))

    executor = UnifiedExecutor(
        llm_judge=llm_judge,
        deterministic_metrics=det,
        aggregator=aggregator
    )

    results = []

    for row in dataset:
        claim = row["claim"]
        evidence = row["evidence"]

        claim_time = datetime.now()  # fallback

        evidence_norm = normalize_evidence(evidence, claim_time)

        entities = resolver.resolve(claim, evidence_norm)

        result = executor.evaluate(
            claim=claim,
            claim_time=claim_time,
            evidence_list=evidence_norm,
            entities=entities
        )

        results.append({
            "claim": claim,
            "label": row.get("label"),
            "final_score": result["final_score"],
            "metrics": result["metrics"],
            "variances": result["variances"]
        })

    # save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results → {output_path}")


# entry point

if __name__ == "__main__":

    aggregator = Aggregator()
    resolver = EntityResolver()

    evaluate_dataset(
        dataset_path="gold_dataset.json",
        output_path="scored_results.json",
        resolver=resolver,
        aggregator=aggregator,
        type_weight_fn=get_type_weight,
        verifiable_fn=is_verifiable
    )