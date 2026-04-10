import re
import json 
from .base_judge import BaseJudge


class DeepSeekJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        response = self.client.chat(
            prompt,
            0.0,     
            500      
        )

        print(self.client.chat("Say hello"))

        print("=== RAW RESPONSE FROM CLIENT ===")
        print(response)

        return self._parse(response)

    def _parse(self, response):
        try:
            text = str(response)  # since Bedrock returns string

            print("RAW TEXT:", text[:500])

            import re
            match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                print("⚠️ NO JSON TAG FOUND")
                return {}

            parsed = json.loads(json_str)

            print("PARSED:", parsed)

            return parsed if isinstance(parsed, dict) else {}

        except Exception as e:
            print("❌ PARSE FAILED:", e)
            return {}