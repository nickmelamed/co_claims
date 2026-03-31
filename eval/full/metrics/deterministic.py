import numpy as np


class DeterministicMetrics:
    def __init__(self, metrics_impl):
        self.metrics = metrics_impl

    def compute(self, claim_f, evidence_entities, domains, n):
        EAS = self.metrics.eas(n)
        Coverage = self.metrics.coverage(claim_f["entities"], evidence_entities)

        SRS = len(set(domains)) / max(1, n)

        EVS = sum(1 for d in domains if "company" not in d) / max(1, n)

        return {
            "EAS": EAS,
            "Coverage": Coverage,
            "SRS": SRS,
            "EVS": EVS
        }