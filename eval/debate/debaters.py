DEBATE_PROMPT_A = """
You are arguing that the claim is TRUE.

Use ONLY the provided evidence. Do NOT use external knowledge.

Claim:
{claim}

Evidence:
{evidence}

Task:
- Construct the strongest possible argument supporting the claim
- Use clear reasoning grounded in the evidence
- If evidence is weak or incomplete, acknowledge limitations

Output ONLY valid JSON:

<json>
{
  "argument": "..."
}
</json>
"""

DEBATE_PROMPT_B = """
You are arguing that the claim is FALSE.

Use ONLY the provided evidence. Do NOT use external knowledge.

Claim:
{claim}

Evidence:
{evidence}

Task:
- Construct the strongest possible argument against the claim
- Identify logical flaws, missing support, or contradictory evidence
- If evidence is weak or incomplete, explain why

Output ONLY valid JSON:

<json>
{
  "argument": "..."
}
</json>
"""


class DebateEngine:
    def __init__(self, pro_model, con_model):
        self.pro_model = pro_model      # Prometheus
        self.con_model = con_model      # Mixtral

    def run(self, claim, evidence_list):
        evidence_text = "\n".join([e["text"] for e in evidence_list])

        pro = self.pro_model.evaluate(
            DEBATE_PROMPT_A.format(claim=claim, evidence=evidence_text)
        )

        con = self.con_model.evaluate(
            DEBATE_PROMPT_B.format(claim=claim, evidence=evidence_text)
        )

        return {
            "pro": pro.get("argument", ""),
            "con": con.get("argument", "")
        }