import json 
from judges.base_judge import BaseJudge

class DeepSeekJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        text = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return self._parse(text)

    def _parse(self, response):
        try:
            text = response.choices[0].message.content
            return json.loads(text.split("<json>")[-1])
        except:
            return {"error": True}