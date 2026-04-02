# TODO need to edit the output based on desired uses 

STRUCTURE_PROMPT = """
Parse the claim into structured components.

Claim:
{claim}

Output JSON:
{{
  "entities": [],
  "metrics": [],
  "direction": "increase/decrease/none",
  "scope": "broad/specific",
  "modality": "strong/hedged",
  "decomposed_claims": []
}}
"""

class ClaimReasoner:
    def __init__(self, judge):
        self.judge = judge

    def structure(self, claim):
        return self.judge.evaluate(
            STRUCTURE_PROMPT.format(claim=claim)
        )
