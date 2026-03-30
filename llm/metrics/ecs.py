from .utils import extract_json, safe_mean

ECS_PROMPT = """
Evaluate how strongly the evidence contradicts the claim.

Claim:
{claim}

Evidence:
{evidence}

Score from 0 to 1:
0 = no contradiction
1 = direct contradiction

Also provide confidence in your judgment.

Output:
<json>
{{"score": float, "confidence": float}}
</json>
"""


class ECS_LLM:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score_evidence(self, claim, evidence):
        result = self.ensemble.evaluate(
        ECS_PROMPT.format(claim=claim, evidence=evidence),
        field="score")
        
        return result["mean"]

    def score(self, claim, evidence_list, relevances):
        scores = []
        weights = []

        for e, r in zip(evidence_list, relevances):
            c = self.score_evidence(claim, e["text"])
            scores.append(c)
            weights.append(r)

        if not scores:
            return 0.0

        return sum(c * w for c, w in zip(scores, weights)) / (sum(weights) + 1e-6)