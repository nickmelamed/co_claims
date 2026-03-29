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

Output:
<json>
{{"score": float}}
</json>
"""


class HLS_LLM:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score(self, claim):
        result = self.ensemble.evaluate(
            HLS_PROMPT.format(claim=claim)
        )

        # using entailment_mean as generic "score"
        return result.get("entailment_mean", 1.0)