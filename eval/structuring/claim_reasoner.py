# TODO need to edit the output based on desired uses 
import re
from datetime import datetime, UTC


DEFAULT_STRUCTURE = {
    "entities": [],
    "metrics": [],
    "direction": None,
    "scope": None,      
    "claim_time": None,
}

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

    def _apply_schema(self, structured: dict) -> dict:
        """Merge LLM output with default schema safely."""
        if not isinstance(structured, dict):
            structured = {}

        # Only keep known keys (prevents LLM junk pollution)
        cleaned = {k: structured.get(k) for k in DEFAULT_STRUCTURE.keys()}

        # Fill defaults
        return {**DEFAULT_STRUCTURE, **cleaned}

    def extract_time(self, claim):
        match = re.search(r"(20\d{2})", claim)
        if match:
            return datetime(int(match.group(1)), 1, 1)

        return datetime.utcnow()

    def structure(self, claim):
        try:
            response = self.judge.evaluate(
                STRUCTURE_PROMPT.format(claim=claim)
            )

            # If evaluate() already returns dict → good
            structured = response

            # If not, try parsing (optional depending on your setup)
            if not isinstance(structured, dict):
                structured = {}

        except Exception:
            structured = {}

        # apply schema 
        structured = self._apply_schema(structured)

        # handle claim time robustly
        try:
            if self.extract_time is not None:
                extracted_time = self.extract_time(claim)
                structured["claim_time"] = extracted_time or datetime.now(UTC)
            else:
                structured["claim_time"] = datetime.now(UTC)

        except Exception:
            structured["claim_time"] = datetime.now(UTC)

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
