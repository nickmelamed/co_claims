class EvidenceTriage:
    def __init__(self, sim, threshold=0.3):
        self.sim = sim
        self.threshold = threshold

    def filter(self, claim_embedding, evidence_list):
        return [
            e for e in evidence_list
            if self.sim.relevance(claim_embedding, e["embedding"]) >= self.threshold
        ]