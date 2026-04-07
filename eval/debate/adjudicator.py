ADJUDICATION_PROMPT = """
You are a strict, neutral judge evaluating two arguments about a claim.

Your task is to compare the arguments and determine:
1. How strongly the FOR argument supports the claim
2. How strongly the AGAINST argument contradicts the claim
3. How confident you are in your evaluation

You MUST base your judgment ONLY on the content of the arguments provided.
Do NOT use external knowledge.

---

Claim:
{claim}

Argument FOR:
{pro}

Argument AGAINST:
{con}

---

SCORING RUBRIC:

support_score (0–1):
- 1.0 = strong, direct, well-reasoned support with clear justification
- 0.5 = partial or weak support, missing justification or clarity
- 0.0 = no meaningful support

contradiction_score (0–1):
- 1.0 = strong, direct contradiction with clear reasoning
- 0.5 = partial or weak contradiction
- 0.0 = no meaningful contradiction

confidence (0–1):
- Reflects how reliable your judgment is based on:
  - clarity of arguments
  - completeness of reasoning
  - absence of ambiguity or conflict
- 1.0 = very clear and decisive
- 0.5 = moderate uncertainty
- 0.0 = highly uncertain or insufficient information

---

IMPORTANT RULES:
- Compare BOTH arguments before scoring
- Do NOT assume the claim is true or false by default
- Do NOT introduce outside facts
- Output ONLY valid JSON (no extra text)

---

Output:
<json>
{{
  "support_score": float,
  "contradiction_score": float,
  "confidence": float
}}
</json>
"""

ADJUDICATION_SCHEMA = {
    "support_score": 0.0,
    "contradiction_score": 0.0,
    "confidence": 0.0,
}


class Adjudicator:
    def __init__(self, judge_model):
        self.judge_model = judge_model  # Prometheus preferred

    def _apply_schema(self, result):
      if not isinstance(result, dict):
          result = {}

      return {
          "support_score": float(result.get("support_score", 0.0)),
          "contradiction_score": float(result.get("contradiction_score", 0.0)),
          "confidence": float(result.get("confidence", 0.0)),
      }
    

    def decide(self, claim, debate_output):
      try:
          result = self.judge_model.evaluate(
              ADJUDICATION_PROMPT.format(
                  claim=claim,
                  pro=debate_output.get("pro", ""),
                  con=debate_output.get("con", "")
              )
          )

      except Exception:
          result = {}

      return self._apply_schema(result)