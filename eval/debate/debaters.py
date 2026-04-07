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

DEBATE_SCHEMA = {
    "argument": ""
}

class DebateEngine:
    def __init__(self, pro_model, con_model):
        self.pro_model = pro_model
        self.con_model = con_model

    def _extract_argument(self, result):
        if not isinstance(result, dict):
            return ""

        arg = result.get("argument", "")
        return arg if isinstance(arg, str) else ""

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

        try:

            pro_task = self._run_pro(claim, evidence_text)
            con_task = self._run_con(claim, evidence_text)

            pro, con = await asyncio.gather(pro_task, con_task)
        
        except Exception:
            pro = {}
            con = {}

        return {
            "pro": pro.get("argument", ""),
            "con": con.get("argument", "")
        }

    def run(self, claim, evidence_list):
        evidence_text = "\n".join([e["text"] for e in evidence_list])

        try:
            pro = self.pro_model.evaluate(
                DEBATE_PROMPT_A.format(claim=claim, evidence=evidence_text)
            )
        except Exception:
            pro = {}

        try:
            con = self.con_model.evaluate(
                DEBATE_PROMPT_B.format(claim=claim, evidence=evidence_text)
            )
        except Exception:
            con = {}

        return {
            "pro": self._extract_argument(pro),
            "con": self._extract_argument(con)
        }