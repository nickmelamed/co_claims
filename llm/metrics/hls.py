from .utils import extract_json

HLS_PROMPT = """
Evaluate the level of hedging in the claim.

Claim:
{claim}

Hedging refers to uncertainty or lack of commitment, such as:
- "may", "might", "could"
- "suggests", "appears"
- vague or non-committal phrasing

Score from 0 to 1:
1 = no hedging (very confident, direct claim)
0 = highly hedged (uncertain, vague claim)

Also provide confidence in your judgment.

Output:
<json>
{{"score": float, "confidence": float}}
</json>
"""


class HLS_LLM:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score(self, claim):
        result = self.ensemble.evaluate(
            HLS_PROMPT.format(claim=claim),
            field="score"
        )
        return result["mean"], result["variance"]