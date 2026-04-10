import json


UNIFIED_PROMPT = """
You are evaluating the credibility of a claim using provided evidence.

Use ONLY the provided evidence. Do not use external knowledge.

Claim:
{claim}

Evidence:
{evidence}

Evaluate the following metrics (0 to 1):

Definitions:
- ESS: Degree to which evidence directly supports the claim.
  0 = no support
  1 = strong, direct support

- ECS: Degree to which evidence contradicts the claim.
  0 = no contradiction
  1 = strong, direct contradiction

- CMS: Degree to which the claim is specific and measurable.
  0 = vague, not testable
  1 = precise, quantifiable, testable

- LCS: Logical consistency of the claim.
  0 = internally contradictory or incoherent
  1 = fully logically consistent

- HLS: Degree of hedging or uncertainty in the claim.
  0 = fully certain (no hedging)
  1 = highly hedged (uncertain, qualified language)

Instructions:
- Treat each piece of evidence independently before aggregating.
- Consider both supporting and contradicting evidence.
- If evidence is insufficient or irrelevant:
  - ESS should be low
  - ECS should be low (unless contradiction is explicit)
  - Confidence should be low
- Evaluate each metric independently.

Scoring guidelines:
0.0 = none
0.25 = weak
0.5 = moderate
0.75 = strong
1.0 = definitive

Confidence reflects:
- amount of evidence
- agreement across evidence
- clarity of support or contradiction

---

Examples:

Example 1:
Claim: "This model improves accuracy by 20% on ImageNet."
Evidence: "The model achieved 20% higher accuracy than baseline on ImageNet in experiments."

<json>
{{
  "ESS": {{"score": 0.95, "confidence": 0.9}},
  "ECS": {{"score": 0.0, "confidence": 0.9}},
  "CMS": {{"score": 1.0, "confidence": 0.95}},
  "LCS": {{"score": 1.0, "confidence": 0.95}},
  "HLS": {{"score": 0.0, "confidence": 0.9}}
}}
</json>

---

Example 2:
Claim: "This model significantly improves performance."
Evidence: "The model showed slight improvements in some cases."

<json>
{{
  "ESS": {{"score": 0.4, "confidence": 0.7}},
  "ECS": {{"score": 0.3, "confidence": 0.7}},
  "CMS": {{"score": 0.2, "confidence": 0.8}},
  "LCS": {{"score": 1.0, "confidence": 0.9}},
  "HLS": {{"score": 0.6, "confidence": 0.8}}
}}
</json>

---

Example 3:
Claim: "The system always returns correct outputs."
Evidence: "The system fails in 30% of edge cases."

<json>
{{
  "ESS": {{"score": 0.0, "confidence": 0.9}},
  "ECS": {{"score": 0.95, "confidence": 0.9}},
  "CMS": {{"score": 0.7, "confidence": 0.8}},
  "LCS": {{"score": 1.0, "confidence": 0.9}},
  "HLS": {{"score": 0.0, "confidence": 0.9}}
}}
</json>

---

Example 4:
Claim: "This approach may improve results under certain conditions."
Evidence: "Some experiments show improvement, but results vary."

<json>
{{
  "ESS": {{"score": 0.5, "confidence": 0.6}},
  "ECS": {{"score": 0.1, "confidence": 0.6}},
  "CMS": {{"score": 0.3, "confidence": 0.7}},
  "LCS": {{"score": 1.0, "confidence": 0.9}},
  "HLS": {{"score": 0.9, "confidence": 0.9}}
}}
</json>

---

Example 5:
Claim: "The algorithm increases speed by 50% and decreases speed under heavy load."
Evidence: "The algorithm increases speed by 50%."

<json>
{{
  "ESS": {{"score": 0.6, "confidence": 0.7}},
  "ECS": {{"score": 0.2, "confidence": 0.6}},
  "CMS": {{"score": 0.9, "confidence": 0.8}},
  "LCS": {{"score": 0.1, "confidence": 0.9}},
  "HLS": {{"score": 0.0, "confidence": 0.9}}
}}
</json>

---

Now evaluate:

<json>
"""

METRICS = ["ESS", "ECS", "CMS", "LCS", "HLS"]

DEFAULT_METRIC = {
    "score": 0.0,
    "confidence": 0.0
}

DEFAULT_SCHEMA = {
    m: DEFAULT_METRIC.copy() for m in METRICS
}


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
        self.metrics = ["ESS", "ECS", "CMS", "LCS", "HLS"]

    async def evaluate(self, claim, evidence_list, relevances):
        final_scores = {m: 0.0 for m in self.metrics}
        final_variances = {m: 0.0 for m in self.metrics}

        weight_sum = sum(relevances) + 1e-6

        for e, r in zip(evidence_list, relevances):
            prompt = UNIFIED_PROMPT.format(
                claim=claim,
                evidence=e["text"]
            )

            try:
                scores, variances, _ = await self.ensemble.evaluate(prompt)

            except Exception:
                scores = {m: 0.0 for m in self.metrics}
                variances = {m: 1.0 for m in self.metrics}

            for m in self.metrics:
                final_scores[m] += scores[m] * r
                final_variances[m] += variances[m]

        # normalize scores
        for m in self.metrics:
            final_scores[m] /= weight_sum
            final_variances[m] /= max(1, len(evidence_list))

        return final_scores, final_variances