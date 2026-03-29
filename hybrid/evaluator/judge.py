import json

JUDGE_PROMPT = """

ROLE: 
You are an expert in claim evaluation. 

TASK: 
You are evaluating whether a piece of evidence supports or contradicts a claim.

CLAIM:
{claim}

EVIDENCE:
{evidence}

INSTRUCTIONS: 
- Determine if the evidence supports, contradicts, or is unrelated to the claim.
- Consider semantic meaning, not just word overlap.
- Detect implicit contradictions, not just explicit negations.
- Be conservative: only assign high scores when strongly justified.

EVALUATION: 
Your evaluation consists of the following: 
- entailment: probability evidence supports claim (0 to 1)
- contradiction: probability evidence contradicts claim (0 to 1)
- confidence: confidence in your judgment (0 to 1)
- rationale: one sentence explanation

OUTPUT: 
Return your evaluation in the following schema: 

    {
        "entailment": 
        "contradiction":
        "confidence": 
        "rationale": 
    }

You MUST only return valid JSON. 

RULES:
- entailment + contradiction ≤ 1
- If the evidence is unrelated to the claim, both entailment and contradiction shold be low
- Only base your evaluation on the provided evidence. Do NOT use any outside information. 

"""

MODEL = "gpt-4o-mini" # placeholder model 

class LLMJudge:
    def __init__(self, client, model=MODEL):
        self.client = client
        self.model = model

    def score(self, claim, evidence):
        prompt = JUDGE_PROMPT.format(
            claim=claim,
            evidence=evidence
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content

        # robust JSON extraction
        try:
            json_str = text.split("<json>")[-1].strip()
            return json.loads(json_str)
        except:
            return {
                "entailment": 0,
                "contradiction": 0,
                "confidence": 0,
                "rationale": "parse_error"
            }