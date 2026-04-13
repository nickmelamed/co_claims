import re
from datetime import datetime, UTC
import json

ENTITY_EXTRACTION_PROMPT = """
Extract key entities from the claim.

Definition:
- Entities = companies, people, products, locations, or measurable concepts

Rules:
- Return ONLY a JSON list
- No explanations
- No duplicates
- Keep entities short and normalized

Claim:
{claim}

Output:
["entity1", "entity2", ...]
"""

class ClaimReasoner:
    def __init__(self, judge):
        self.judge = judge

    def _safe_parse_list(self, text):
        try:
            return json.loads(text)
        except:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    return []
        return []

    def extract_entities(self, claim):
        try:
            response = self.judge.evaluate(
                ENTITY_EXTRACTION_PROMPT.format(claim=claim)
            )

            # Handle cases where model returns string
            if isinstance(response, str):
                entities = self._safe_parse_list(response)
            else:
                entities = response

            if not isinstance(entities, list):
                return []

            # normalize
            return [
                e.strip().lower()
                for e in entities
                if isinstance(e, str) and len(e.strip()) > 0
            ]

        except Exception:
            return []

    def extract_time(self, claim):
        match = re.search(r"(20\d{2})", claim)
        if match:
            return datetime(int(match.group(1)), 1, 1)

        return datetime.now(UTC)
    
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
