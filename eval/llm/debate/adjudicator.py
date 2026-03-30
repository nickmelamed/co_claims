ADJUDICATION_PROMPT = """
You are a neutral judge evaluating two arguments.

Claim:
{claim}

Argument FOR:
{pro}

Argument AGAINST:
{con}

Decide:

- support_score (0–1)
- contradiction_score (0–1)
- confidence (0–1)

Output:
<json>
{{
  "support_score": float,
  "contradiction_score": float,
  "confidence": float
}}
</json>
"""


class Adjudicator:
    def __init__(self, judge_model):
        self.judge_model = judge_model  # Prometheus preferred

    def decide(self, claim, debate_output):
        result = self.judge_model.evaluate(
            ADJUDICATION_PROMPT.format(
                claim=claim,
                pro=debate_output["pro"],
                con=debate_output["con"]
            )
        )

        return result