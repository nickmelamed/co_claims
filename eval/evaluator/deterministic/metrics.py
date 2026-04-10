import math
import numpy as np

EPS = 1e-6
K_EVIDENCE = 5

class DeterministicMetrics:
    def __init__(
        self,
        k_evidence: int = 5,
        half_life: float = 365.0,
        epsilon: float = 1e-6,
        type_weight_fn=None,
        verifiable_fn=None,
        time_fn=None,
        normalize_scores: bool = True,
    ):
        self.K_EVIDENCE = k_evidence
        self.HALF_LIFE = half_life
        self.EPS = epsilon

        # from source_types.py
        self.type_weight_fn = type_weight_fn
        self.verifiable_fn = verifiable_fn

        self.time_fn = time_fn
        self.normalize_scores = normalize_scores

    def weighted_avg(self, values, weights):
            if not values:
                return 0
            return np.sum(np.array(values) * np.array(weights)) / (np.sum(weights) + EPS)

    def _clip(self, x):
        return max(0.0, min(1.0, x))

    def cscope(self, entities):
            n = len(entities)
            return 1 / (1 + math.log(1 + n))

    def eas(self, n):
            return 1 - np.exp(-n / K_EVIDENCE)

    def ers(self, claim_time, evidence_times, half_life=90):
        if not evidence_times:
            return 0
    
        tau = half_life * 84600 # convert days to seconds 
        
        scores = []
        for t in evidence_times:
             delta = (claim_time - t).total_seconds()
             delta = max(0, delta)

             score = np.exp(-delta * np.log(2) / tau)
             scores.append(score)
        return np.mean(scores)

    def ests(self, relevances, source_types):
        if not relevances or not source_types:
            return 0

        weights = [self.type_weight_fn(t) for t in source_types]

        # base score
        num = sum(r * w for r, w in zip(relevances, weights))
        denom = sum(relevances) + self.EPS
        base_score = num / denom

        # coverage = how many sources are meaningfully typed
        coverage = sum(1 for t in source_types if t != "unknown") / len(source_types)

        return self._clip(base_score * coverage)

    def eags(self, supports):
        if not supports:
            return 0

        mean = np.mean(supports)
        variance = np.mean([(s - mean) ** 2 for s in supports])

        return 1 - variance

    def srs(self, domains):
        if not domains:
            return 0

        valid_domains = [d for d in domains if d and d != "unknown"]

        if not valid_domains:
            return 0

        diversity = len(set(valid_domains)) / len(valid_domains)

        # coverage = how many domains are valid
        coverage = len(valid_domains) / len(domains)

        return self._clip(diversity * coverage)

    def evs(self, source_types):
        if not source_types:
            return 0

        flags = [self.verifiable_fn(s) for s in source_types]

        # base score
        base_score = sum(flags) / len(flags)

        # coverage = how many sources are classified
        coverage = sum(1 for s in source_types if s != "unknown") / len(source_types)

        return self._clip(base_score * coverage)
