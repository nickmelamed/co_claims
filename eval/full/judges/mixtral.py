import json 
from judges.base_judge import BaseJudge

class MixtralJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        response = self.client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-v0.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3 # higher than prometheus for variety
        )
        return self._parse(response)

    def _parse(self, response):
        try:
            text = response.choices[0].message.content
            return json.loads(text.split("<json>")[-1])
        except:
            return {"error": True}