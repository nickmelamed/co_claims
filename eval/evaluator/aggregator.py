GAMMA = 5

class Aggregator:
    def credibility(self, evidence_score, claim_score, n):
        weight = n / (n + GAMMA)
        return weight * evidence_score + (1 - weight) * claim_score