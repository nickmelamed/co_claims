import json


UNIFIED_PROMPT = """
You are evaluating the credibility of a claim using provided evidence.

Claim:
{claim}

Evidence:
{evidence}

Evaluate the following metrics (0 to 1):

1. ESS (Evidence Support Score)
- How strongly does the evidence support the claim?

2. ECS (Evidence Contradiction Score)
- How strongly does the evidence contradict the claim?

3. CMS (Claim Measurability Score)
- Is the claim testable and quantifiable?

4. LCS (Logical Consistency Score)
- Is the claim internally consistent?

5. HLS (Hedging Level Score)
- Is the claim direct and confident (not hedged)?

Output:
<json>
{{
  "ESS": float,
  "ECS": float,
  "CMS": float,
  "LCS": float,
  "HLS": float,
  "confidence": float
}}
</json>
"""


def extract_json(text):
    try:
        if "<json>" in text:
            text = text.split("<json>")[-1]
        return json.loads(text.strip())
    except:
        return {"error": True}


class UnifiedLLMJudge:
    def __init__(self, model):
        self.model = model

    def evaluate(self, claim, evidence_text):
        prompt = UNIFIED_PROMPT.format(
            claim=claim,
            evidence=evidence_text
        )

        response = self.model.evaluate(prompt)

        parsed = extract_json(response)

        return parsed if "error" not in parsed else {
            "ESS": 0,
            "ECS": 0,
            "CMS": 0,
            "LCS": 1,
            "HLS": 1,
            "confidence": 0
        }