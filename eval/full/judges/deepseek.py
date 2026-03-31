import json 
from judges.base_judge import BaseJudge

class DeepSeekJudge(BaseJudge):
    def __init__(self, client):
        self.client = client

    def evaluate(self, prompt):
        response = self.client.chat.completions.create(
            model="deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return self._parse(response)

    def _parse(self, response):
        try:
            text = response.choices[0].message.content
            return json.loads(text.split("<json>")[-1])
        except:
            return {"error": True}