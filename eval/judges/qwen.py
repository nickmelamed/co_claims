import re
import json 
from .base_judge import BaseJudge

class QwenJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        response = self.client.chat(
            prompt,
            0.0,    
            500     
        )
        return self._parse(response)

    def _parse(self, response):
        try:
            text = str(response)

            # 1. Try <json> tags
            match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            # 2. Fallback: try raw JSON extraction
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))

            print("⚠️ NO JSON FOUND AT ALL")
            return {}

        except Exception as e:
            print("❌ PARSE FAILED:", e)
            return {}