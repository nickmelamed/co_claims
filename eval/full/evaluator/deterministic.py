import numpy as np

# constants 

K_EVIDENCE = 5


class DeterministicMetrics:
    def __init__(self, metrics_impl):
        self.metrics = metrics_impl

    def eas(self, n):
        return 1 - np.exp(-n / K_EVIDENCE)
    
    def coverage(self, claim_entities, evidence_entities):
        union = set().union(*evidence_entities) if evidence_entities else set()
        return len(claim_entities & union) / max(1, len(claim_entities))
    
    

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