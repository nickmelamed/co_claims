import numpy as np

class EvidenceTriage:
    def __init__(self, sim, threshold=0.2, top_k=5):
        self.sim = sim
        self.threshold = threshold
        self.top_k = top_k

    def _score(self, claim_embedding, evidence_list):
        scored = []
        for e in evidence_list:
            score = self.sim.relevance(claim_embedding, e["embedding"])
            scored.append((e, score))
        return scored

    def filter(self, claim_embedding, evidence_list):
        if not evidence_list:
            return []

        # score
        scored = self._score(claim_embedding, evidence_list)

        # sort (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # take top_k
        top_candidates = scored[: self.top_k]

        # apply threshold
        filtered = [
            e for e, s in top_candidates
            if s >= self.threshold
        ]

        # Safe fallback (guarantee ≥1 evidence)
        if not filtered:
            filtered = [top_candidates[0][0]]

        return filtered