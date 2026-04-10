import sys
import os
import asyncio
import json
from datetime import datetime
from functools import partial

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from dotenv import load_dotenv
load_dotenv()

import cohere

from eval.evaluator.llm.metrics import UNIFIED_PROMPT
from eval.evaluator.deterministic.metrics import DeterministicMetrics
from eval.evaluator.executor import UnifiedExecutor
from eval.evaluator.aggregator import Aggregator
from eval.evaluator.deterministic.source_types import get_type_weight, is_verifiable, extract_domain
from eval.structuring.entity_resolver import EntityResolver
from eval.evaluator.deterministic.extractor import FeatureExtractor

import hashlib
from pathlib import Path

# config
MAX_EVIDENCE = 3
MAX_CHARS = 300
CONCURRENCY_LIMIT = 3
RETRIES = 3

# cache 
class LLMCache:
    def __init__(self, path=".llm_cache.json"):
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

    def _key(self, claim, evidence):
        raw = claim + "||" + evidence
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, claim, evidence):
        return self.cache.get(self._key(claim, evidence))

    def set(self, claim, evidence, value):
        self.cache[self._key(claim, evidence)] = value


# cohere judge 
class CohereJudge:
    def __init__(self, api_key, model="command-a-03-2025"):
        if not api_key:
            raise ValueError("CO_API_KEY not set")

        self.client = cohere.Client(api_key)
        self.model = model
        self.cache = LLMCache()

    async def evaluate(self, claim, evidence_list, relevances):
        loop = asyncio.get_event_loop()

        metrics = ["ESS", "ECS", "CMS", "LCS", "HLS"]

        final_scores = {m: 0.0 for m in metrics}
        final_variances = {m: 0.0 for m in metrics}

        # reduce evidence count
        evidence_list = sorted(
            evidence_list,
            key=lambda x: x.get("relevance", 0.5),
            reverse=True
        )[:MAX_EVIDENCE]

        relevances = [e.get("relevance", 0.5) for e in evidence_list]

        weight_sum = sum(relevances) + 1e-6
        if weight_sum <= 1e-6:
            weight_sum = len(evidence_list) or 1
            relevances = [1.0] * len(evidence_list)

        # parallelize per-evidence calls
        async def eval_single(e):
            evidence_text = e["text"][:MAX_CHARS]

            return await self._call_with_retry(
                loop,
                claim,
                evidence_text
            )

        tasks = [eval_single(e) for e in evidence_list]
        responses = await asyncio.gather(*tasks)

        for e, r, response in zip(evidence_list, relevances, responses):
            parsed = self._parse(response.text)
            if not parsed:
                continue

            for m in metrics:
                score = parsed.get(m, {}).get("score", 0.0)
                confidence = parsed.get(m, {}).get("confidence", 0.0)

                final_scores[m] += score * r
                final_variances[m] += (1 - confidence)

        # normalize
        for m in metrics:
            final_scores[m] /= weight_sum
            final_variances[m] /= max(1, len(evidence_list))

        return final_scores, final_variances, responses

    async def _call_with_retry(self, loop, claim, evidence):
        cached = self.cache.get(claim, evidence)
        if cached:
            return cached

        prompt = UNIFIED_PROMPT.format(
            claim=claim,
            evidence=evidence
        )

        for i in range(RETRIES):
            try:
                response = await loop.run_in_executor(
                    None,
                    partial(
                        self.client.chat,
                        model=self.model,
                        message=prompt,
                        temperature=0.0
                    )
                )

                # save to cache
                self.cache.set(claim, evidence, response)
                return response

            except Exception as e:
                if i == RETRIES - 1:
                    raise e
                await asyncio.sleep(2 ** i)

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


# helpers
def parse_time_safe(date_str):
    if not date_str or date_str != date_str:
        return datetime.now()
    try:
        return datetime.fromisoformat(date_str)
    except:
        return datetime.now()


def normalize_evidence(evidence_list):
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
async def evaluate_dataset(
    dataset_path,
    output_path,
    resolver,
    aggregator,
    type_weight_fn,
    verifiable_fn
):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    det = DeterministicMetrics(
        type_weight_fn=type_weight_fn,
        verifiable_fn=verifiable_fn
    )

    llm_judge = CohereJudge(os.getenv("CO_API_KEY"))

    executor = UnifiedExecutor(
        llm_judge=llm_judge,
        deterministic_metrics=det,
        aggregator=aggregator
    )

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_row(row):
        async with sem:
            claim = row["claim"]
            evidence = row["evidence"]

            claim_time = datetime.now()
            evidence_norm = normalize_evidence(evidence)

            entities = resolver.resolve(claim, evidence_norm)

            result = await executor.evaluate(
                claim=claim,
                claim_time=claim_time,
                evidence_list=evidence_norm,
                entities=entities
            )

            return {
                "claim": claim,
                "label": row.get("label"),
                "final_score": result["final_score"],
                "metrics": result["metrics"],
                "variances": result["variances"]
            }

    tasks = [process_row(row) for row in dataset]
    results = await asyncio.gather(*tasks)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results → {output_path}")

    llm_judge.cache.save()


# entry 
if __name__ == "__main__":
    extractor = FeatureExtractor()
    aggregator = Aggregator()
    resolver = EntityResolver(extractor)

    asyncio.run(
        evaluate_dataset(
            dataset_path="./gold_dataset.json",
            output_path="./gold_scores.json",
            resolver=resolver,
            aggregator=aggregator,
            type_weight_fn=get_type_weight,
            verifiable_fn=is_verifiable
        )
    )