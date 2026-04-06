import asyncio

DEBATE_PROMPT_A = """
You are arguing that the claim is TRUE.

Use ONLY the provided evidence. Do NOT use external knowledge.

Claim:
{claim}

Evidence:
{evidence}

Output ONLY valid JSON:

<json>
{{
  "argument": "..."
}}
</json>
"""

DEBATE_PROMPT_B = """
You are arguing that the claim is FALSE.

Use ONLY the provided evidence. Do NOT use external knowledge.

Claim:
{claim}

Evidence:
{evidence}

Output ONLY valid JSON:

<json>
{{
  "argument": "..."
}}
</json>
"""


class DebateEngine:
    def __init__(self, pro_model, con_model):
        self.pro_model = pro_model
        self.con_model = con_model

    async def _run_pro(self, claim, evidence_text):
        return await asyncio.to_thread(
            self.pro_model.evaluate,
            DEBATE_PROMPT_A.format(claim=claim, evidence=evidence_text)
        )

    async def _run_con(self, claim, evidence_text):
        return await asyncio.to_thread(
            self.con_model.evaluate,
            DEBATE_PROMPT_B.format(claim=claim, evidence=evidence_text)
        )

    async def run_async(self, claim, evidence_list):
        evidence_text = "\n".join([e["text"] for e in evidence_list])

        pro_task = self._run_pro(claim, evidence_text)
        con_task = self._run_con(claim, evidence_text)

        pro, con = await asyncio.gather(pro_task, con_task)

        return {
            "pro": pro.get("argument", ""),
            "con": con.get("argument", "")
        }

    def run(self, claim, evidence_list):
        """
        Backward-compatible sync version
        """
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