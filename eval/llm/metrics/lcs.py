from .utils import extract_json

LCS_PROMPT = """
Evaluate the logical consistency of the claim.

Claim:
{claim}

Instructions:
- Break the claim into sub-statements if needed
- Identify contradictions or inconsistencies within the claim
- Consider implicit contradictions

Score from 0 to 1:
1 = fully logically consistent
0 = internally contradictory

Also provide confidence in your judgment.

Output:
<json>
{{"score": float, "confidence": float}}
</json>
"""


class LCS_LLM:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score(self, claim):
        result = self.ensemble.evaluate(
            LCS_PROMPT.format(claim=claim),
            field="score"
        )
        return result["mean"], result["variance"]