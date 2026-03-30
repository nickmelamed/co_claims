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

Also provide confidence in your judgment.

Output:
<json>
{{"score": float, "confidence": float}}
</json>
"""


class CMS_LLM:
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def score(self, claim):
        result = self.ensemble.evaluate(
            CMS_PROMPT.format(claim=claim),
            field="score"
        )
        return result["mean"], result["variance"]