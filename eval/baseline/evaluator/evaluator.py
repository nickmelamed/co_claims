from urllib.parse import urlparse

from .extractor import FeatureExtractor
from .similarity import Similarity
from .support import SupportScorer
from .contradiction import ContradictionScorer
from .metrics import Metrics
from .aggregator import Aggregator

TAU_R = 0.3

class ClaimEvaluator:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.extractor = FeatureExtractor()
        self.sim = Similarity()
        self.support = SupportScorer()
        self.contradiction = ContradictionScorer()
        self.metrics = Metrics()
        self.aggregator = Aggregator()

    def evaluate(self, claim, evidence_list):
        claim_f = self.extractor.extract(claim)
        claim_embedding = self.embed_fn(claim)

        relevances, supports, contradictions = [], [], []
        evidence_entities, domains, external_flags = [], [], []

        for e in evidence_list:
            ef = self.extractor.extract(e["text"])
            r = self.sim.relevance(claim_embedding, e["embedding"])

            if r < TAU_R:
                continue

            s = self.support.score(claim_f, ef)
            c = self.contradiction.score(claim_f, ef)
            s, c = self.metrics.normalize(s, c)

            relevances.append(r)
            supports.append(s)
            contradictions.append(c)
            evidence_entities.append(ef["entities"])

            domain = urlparse(e["source"]).netloc
            domains.append(domain)
            external_flags.append(int("company" not in domain))

        n = len(supports)

        ESS = self.metrics.ess(supports, relevances)
        ECS = self.metrics.ecs(contradictions, relevances)
        EAS = self.metrics.eas(n)
        Coverage = self.metrics.coverage(claim_f["entities"], evidence_entities)

        CMS = self.metrics.cms(claim_f["entities"])
        Uncertainty = self.metrics.uncertainty(n)

        evidence_score = (ESS + ECS + EAS) / 3
        claim_score = CMS

        credibility = self.aggregator.credibility(evidence_score, claim_score, n)

        return {
            "ESS": ESS,
            "ECS": ECS,
            "EAS": EAS,
            "Coverage": Coverage,
            "CMS": CMS,
            "Uncertainty": Uncertainty,
            "Credibility": credibility
        }