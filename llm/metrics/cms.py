from .utils import extract_json

CMS_PROMPT = """
Evaluate how measurable the claim is.

Claim:
{claim}

Criteria:
- Are entities quantifiable?
- Are metrics clearly defined?
- Is the claim testable?

Score from 0 to 1:
0 = not measurable
1 = fully measurable

Output:
<json>
{{"score": float}}
</json>
"""


class CMS_LLM:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score(self, claim):
        result = self.ensemble.evaluate(
            CMS_PROMPT.format(claim=claim)
        )
        return result.get("entailment_mean", 0.0)