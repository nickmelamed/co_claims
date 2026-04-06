import json 
from judges.base_judge import BaseJudge

class PrometheusJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        text = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return self._parse(text)

    def _parse(self, text):
        try:
            return json.loads(text.split("<json>")[-1])
        except:
            return {"error": True, "raw": text}