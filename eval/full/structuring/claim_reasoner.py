# TODO need to edit the output based on desired uses 
import re
from datetime import datetime

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

    def extract_time(self, claim):
        match = re.search(r"(20\d{2})", claim)
        if match:
            return datetime(int(match.group(1)), 1, 1)

        return datetime.utcnow()

    def structure(self, claim):
        structured = self.judge.evaluate(
            STRUCTURE_PROMPT.format(claim=claim)
        )

        # inject deterministic time
        structured["claim_time"] = self.extract_time(claim)

        return structured
    
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
