from .utils import extract_json, safe_mean
import numpy as np

ESS_PROMPT = """
Evaluate how strongly the evidence supports the claim.

Claim:
{claim}

Evidence:
{evidence}

Score from 0 to 1:
0 = no support
1 = strong direct support

Also provide confidence in your judgment.

Output:
<json>
{{"score": float, "confidence": float}}
</json>
"""


class ESS_LLM:
    
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score_evidence(self, claim, evidence):
        result = self.ensemble.evaluate(
            ESS_PROMPT.format(claim=claim, evidence=evidence),
            field="score"
        )
        return result

    def score(self, claim, evidence_list, relevances):
        scores = []
        weights = []
        variances = []

        for e, r in zip(evidence_list, relevances):
            res = self.score_evidence(claim, e["text"])

            scores.append(res["mean"])
            weights.append(r)
            variances.append(res["variance"])

        if not scores:
            return 0.0, 1.0

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / (sum(weights) + 1e-6)

        return weighted_score, np.mean(variances)