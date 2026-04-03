DEBATE_PROMPT_A = """
You are arguing that the claim is TRUE.

Claim:
{claim}

Evidence:
{evidence}

Provide a concise argument supporting the claim.

Output:
<json>
{{"argument": "..."}}
</json>
"""

DEBATE_PROMPT_B = """
You are arguing that the claim is FALSE.

Claim:
{claim}

Evidence:
{evidence}

Critique the claim and identify flaws.

Output:
<json>
{{"argument": "..."}}
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