import numpy as np

EPS = 1e-6
K_EVIDENCE = 5

METRICS = {"accuracy", "precision", "recall", "latency", "throughput", "f1"}
BENCHMARKS = {"imagenet", "glue", "mnist", "cifar"}

class Metrics:

    def normalize(self, sup, con):
        denom = sup + con + EPS
        return sup / denom, con / denom

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

    def coverage(self, claim_entities, evidence_entities):
        union = set().union(*evidence_entities) if evidence_entities else set()
        return len(claim_entities & union) / max(1, len(claim_entities))

    def cms(self, entities):
        weights = []
        for e in entities:
            if any(char.isdigit() for char in e):
                weights.append(1.0)
            elif any(m in e for m in METRICS):
                weights.append(0.8)
            elif any(b in e for b in BENCHMARKS):
                weights.append(0.6)
            else:
                weights.append(0)
        return sum(weights) / max(1, len(entities))

    def uncertainty(self, n):
        return 1 / (1 + n)