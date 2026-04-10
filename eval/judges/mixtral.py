import re
import json 
from .base_judge import BaseJudge

class MixtralJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return self._parse(response)

    def _parse(self, response):
        try:
            # Handle multiple response formats safely
            if hasattr(response, "choices"):  # OpenAI-style
                text = response.choices[0].message.content
            elif isinstance(response, dict):
                # Bedrock-style
                text = response.get("output", {}) \
                            .get("message", {}) \
                            .get("content", [{}])[0] \
                            .get("text", "")
            else:
                text = str(response)

            # Extract JSON safely
            match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                json_str = text.strip()

            parsed = json.loads(json_str)

            if not isinstance(parsed, dict):
                return {}

            return parsed

        except Exception as e:
            print("MIXTRAL PARSE FAILED:", e)
            print("RAW TEXT:", text[:500])
            return {}