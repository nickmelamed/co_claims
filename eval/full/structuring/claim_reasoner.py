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
    
    def rephrase(self, claim):
        prompt = """
        Rewrite this claim to improve clarity.

        Requirements:
        - Remove vague or hedging language (e.g., "may", "could", "often")
        - Make it more specific and measurable if possible
        - Preserve original meaning

        Claim: {claim}

        Return only the rewritten claim.
        """

        return self.judge.evaluate(prompt.format(claim=claim))
