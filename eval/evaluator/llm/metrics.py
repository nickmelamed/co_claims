import json


UNIFIED_PROMPT = """
You are evaluating the credibility of a claim using provided evidence.

Claim:
{claim}

Evidence:
{evidence}

Evaluate the following metrics (0 to 1):

1. ESS (Evidence Support Score)
2. ECS (Evidence Contradiction Score)
3. CMS (Claim Measurability Score)
4. LCS (Logical Consistency Score)
5. HLS (Hedging Level Score)

Output:
<json>
{{
  "ESS": {{"score": float, "confidence": float}},
  "ECS": {{"score": float, "confidence": float}},
  "CMS": {{"score": float, "confidence": float}},
  "LCS": {{"score": float, "confidence": float}},
  "HLS": {{"score": float, "confidence": float}}
}}
</json>
"""


def extract_json(text):
    try:
        if "<json>" in text:
            text = text.split("<json>")[-1]

        parsed = json.loads(text.strip())

        # validate structure
        if not isinstance(parsed, dict):
            return None

        return parsed

    except Exception:
        return None


class UnifiedLLMJudge:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def evaluate_single(self, claim, evidence_text, field):
        prompt = UNIFIED_PROMPT.format(
            claim=claim,
            evidence=evidence_text
        )

        return self.ensemble.evaluate(prompt, field=field)

    def evaluate(self, claim, evidence_list, relevances):
        metrics = ["ESS", "ECS", "CMS", "LCS", "HLS"]

        final_scores = {m: 0.0 for m in metrics}
        final_variances = {m: 0.0 for m in metrics}

        weight_sum = sum(relevances) + 1e-6

        for e, r in zip(evidence_list, relevances):
            prompt = UNIFIED_PROMPT.format(
                claim=claim,
                evidence=e["text"]
            )

            scores, variances, raw = self.ensemble.evaluate(prompt)

            for m in metrics:
                final_scores[m] += scores.get(m, 0.0) * r
                final_variances[m] += variances.get(m, 0.0)

        # normalize
        for m in metrics:
            final_scores[m] /= weight_sum
            final_variances[m] /= max(1, len(evidence_list))

        return final_scores, final_variances, raw