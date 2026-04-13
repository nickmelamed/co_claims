import json
import numpy as np
import re

UNIFIED_PROMPT = """
You are evaluating the credibility of a claim using provided evidence.

Use ONLY the provided evidence for ESS and ECS.
Evaluate CMS, LCS, and HLS based on the claim itself.

Claim:
{claim}

Evidence:
{evidence}

Evaluate the following metrics (0 to 1):

Definitions:
- ESS: Degree to which evidence directly supports the claim.
- ECS: Degree to which evidence contradicts the claim.
- CMS: Degree to which the claim is specific and measurable.
- LCS: Logical consistency of the claim.
- HLS: Degree of hedging or uncertainty in the claim.

Critical Rules:
- ESS and ECS must NOT both be high:
  - If evidence strongly supports → ECS ≈ 0
  - If evidence strongly contradicts → ESS ≈ 0
- If evidence is irrelevant:
  - ESS = 0.0–0.1
  - ECS = 0.0–0.1
- If evidence is weakly related:
  - Reduce ESS/ECS and confidence accordingly
- If the evidence does not explicitly mention the same entity (e.g., Apple),
then ESS must be ≤ 0.2.
- Do NOT infer beyond evidence

Scoring:
0.0 = none
0.25 = weak
0.5 = moderate
0.75 = strong
1.0 = definitive

Confidence reflects:
- evidence quality
- clarity of signal
- ambiguity

---

Examples:

Example 1:
Claim: "This model improves accuracy by 20% on ImageNet."
Evidence: "The model achieved 20% higher accuracy than baseline on ImageNet in experiments."

<json>
{{
  "ESS": {{"score": 0.95, "confidence": 0.9}},
  "ECS": {{"score": 0.0, "confidence": 0.9}},
  "CMS": {{"score": 1.0, "confidence": 0.95}},
  "LCS": {{"score": 1.0, "confidence": 0.95}},
  "HLS": {{"score": 0.0, "confidence": 0.9}}
}}
</json>

---

Example 2:
Claim: "This model significantly improves performance."
Evidence: "The model showed slight improvements in some cases."

<json>
{{
  "ESS": {{"score": 0.4, "confidence": 0.7}},
  "ECS": {{"score": 0.3, "confidence": 0.7}},
  "CMS": {{"score": 0.2, "confidence": 0.8}},
  "LCS": {{"score": 1.0, "confidence": 0.9}},
  "HLS": {{"score": 0.6, "confidence": 0.8}}
}}
</json>

---

Example 3:
Claim: "The system always returns correct outputs."
Evidence: "The system fails in 30% of edge cases."

<json>
{{
  "ESS": {{"score": 0.0, "confidence": 0.9}},
  "ECS": {{"score": 0.95, "confidence": 0.9}},
  "CMS": {{"score": 0.7, "confidence": 0.8}},
  "LCS": {{"score": 1.0, "confidence": 0.9}},
  "HLS": {{"score": 0.0, "confidence": 0.9}}
}}
</json>

---

Example 4:
Claim: "This approach may improve results under certain conditions."
Evidence: "Some experiments show improvement, but results vary."

<json>
{{
  "ESS": {{"score": 0.5, "confidence": 0.6}},
  "ECS": {{"score": 0.1, "confidence": 0.6}},
  "CMS": {{"score": 0.3, "confidence": 0.7}},
  "LCS": {{"score": 1.0, "confidence": 0.9}},
  "HLS": {{"score": 0.9, "confidence": 0.9}}
}}
</json>

---

Example 5:
Claim: "The algorithm increases speed by 50% and decreases speed under heavy load."
Evidence: "The algorithm increases speed by 50%."

<json>
{{
  "ESS": {{"score": 0.6, "confidence": 0.7}},
  "ECS": {{"score": 0.2, "confidence": 0.6}},
  "CMS": {{"score": 0.9, "confidence": 0.8}},
  "LCS": {{"score": 0.1, "confidence": 0.9}},
  "HLS": {{"score": 0.0, "confidence": 0.9}}
}}
</json>

---

Now evaluate. Output ONLY valid JSON inside <json>...</json> tags. Do not include any extra text. 

<json>
{{
  "ESS": {{"score": float, "confidence": float}},
  "ECS": {{"score": float, "confidence": float}},
  "CMS": {{"score": float, "confidence": float}},
  "LCS": {{"score": float, "confidence": float}},
  "HLS": {{"score": float, "confidence": float}}
}}

IMPORTANT:
- Do NOT include any text outside <json> tags
- Do NOT explain anything
- Do NOT include markdown
- ONLY output the JSON block

Return EXACTLY this format: 
</json>
{{
  "ESS": {{"score": float, "confidence": float}},
  "ECS": {{"score": float, "confidence": float}},
  "CMS": {{"score": float, "confidence": float}},
  "LCS": {{"score": float, "confidence": float}},
  "HLS": {{"score": float, "confidence": float}}
}}
</json>
"""

METRICS = ["ESS", "ECS", "CMS", "LCS", "HLS"]

DEFAULT_METRIC = {
    "score": 0.0,
    "confidence": 0.0
}

DEFAULT_SCHEMA = {
    m: DEFAULT_METRIC.copy() for m in METRICS
}


def extract_json(text):
    try:
        # Extract between <json>...</json>
        match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # fallback: try raw text
            text = text.strip()

        return json.loads(text)

    except Exception as e:
        print("JSON PARSE ERROR:", e)
        #print("RAW TEXT:", text[:500])
        return None


class UnifiedLLMJudge:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.metrics = ["ESS", "ECS", "CMS", "LCS", "HLS"]

    async def evaluate(self, claim, evidence_list, relevances):
        final_scores = {m: 0.0 for m in self.metrics}
        final_weights = {m: 0.0 for m in self.metrics}
        final_variances = {m: 0.0 for m in self.metrics}

        per_evidence_scores = []

        for e, r in zip(evidence_list, relevances):
            prompt = UNIFIED_PROMPT.format(
                claim=claim,
                evidence=e["text"]
            )

            try:
                scores, variances, raw = await self.ensemble.evaluate_async(prompt)
            except Exception as e:
                scores = {m: 0.0 for m in self.metrics}
                variances = {m: 1.0 for m in self.metrics}
                raw = []

                # debugging 
                print("🚨 LLM CALL FAILED:", str(e), flush=True)

            per_evidence_scores.append({
                m: {
                    "score": scores[m],
                    "confidence": np.mean([
                        o[m]["confidence"] for o in raw
                    ]) if raw else 0.0
                }
                for m in self.metrics
            })

            # confidence-weighted aggregation
            for m in self.metrics:
                s = scores[m]
                c = max(0.3, per_evidence_scores[-1][m]["confidence"])

                weight = max(c, 0.2) * max(r, 0.2)

                final_scores[m] += s * weight
                final_weights[m] += weight

                # variance accumulation (confidence-weighted)
                final_variances[m] += variances[m] * weight

        # normalize
        for m in self.metrics:
          if final_weights[m] > 1e-6:   # want a little over zero for stability
              final_scores[m] /= final_weights[m]
              final_variances[m] /= final_weights[m]
          else:
              # fallback to unweighted mean if weights are unusable
              vals = [
                  o[m]["score"] for o in raw
                  if isinstance(o.get(m), dict)
              ]

              if vals:
                  final_scores[m] = float(np.mean(vals))
              else:
                  final_scores[m] = 0.0

              final_variances[m] = 1.0

        return final_scores, final_variances, per_evidence_scores