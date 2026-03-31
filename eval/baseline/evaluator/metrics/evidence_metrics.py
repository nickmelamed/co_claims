import numpy as np

EPS = 1e-6
K_EVIDENCE = 5


class EvidenceMetrics:

    def weighted_avg(self, values, weights):
        if not values:
            return 0
        return np.sum(np.array(values) * np.array(weights)) / (np.sum(weights) + EPS)

    def ess(self, supports, relevances):
        return self.weighted_avg(supports, relevances)

    def ecs(self, contradictions, relevances):
        return self.weighted_avg(contradictions, relevances)

    def eas(self, n):
        return 1 - np.exp(-n / K_EVIDENCE)

    def ers(self, claim_time, evidence_times, half_life=365):
        if not evidence_times:
            return 0

        lmbda = np.log(2) / half_life
        scores = [
            np.exp(-lmbda * max(0, (claim_time - t)))
            for t in evidence_times
        ]
        return np.mean(scores)

    def ests(self, relevances, type_weights):
        if not relevances:
            return 0

        num = sum(r * t for r, t in zip(relevances, type_weights))
        denom = sum(relevances) + EPS
        return num / denom

    def eags(self, supports):
        if not supports:
            return 0

        mean = np.mean(supports)
        variance = np.mean([(s - mean) ** 2 for s in supports])

        return 1 - variance