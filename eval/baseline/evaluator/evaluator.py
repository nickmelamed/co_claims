from urllib.parse import urlparse

from .extractor import FeatureExtractor
from .similarity import Similarity
from .support import SupportScorer
from .contradiction import ContradictionScorer
from .aggregator import Aggregator

from .metrics import EvidenceMetrics, SourceMetrics, ClaimMetrics

TAU_R = 0.3


class ClaimEvaluator:
    def __init__(self, embed_fn):
        self.embed_fn = embed_fn

        self.extractor = FeatureExtractor()
        self.sim = Similarity()
        self.support = SupportScorer()
        self.contradiction = ContradictionScorer()

        self.evidence_metrics = EvidenceMetrics()
        self.source_metrics = SourceMetrics()
        self.claim_metrics = ClaimMetrics(self.contradiction)

        self.aggregator = Aggregator()

    def evaluate(self, claim, evidence_list, claim_time=0):
        claim_f = self.extractor.extract(claim)
        claim_embedding = self.embed_fn(claim)

        relevances, supports, contradictions = [], [], []
        domains, external_flags = [], []
        type_weights, timestamps = [], []

        for e in evidence_list:
            ef = self.extractor.extract(e["text"])
            r = self.sim.relevance(claim_embedding, e["embedding"])

            if r < TAU_R:
                continue

            s = self.support.score(claim_f, ef)
            c = self.contradiction.score(claim_f, ef)

            denom = s + c + 1e-6
            s, c = s / denom, c / denom

            relevances.append(r)
            supports.append(s)
            contradictions.append(c)

            domains.append(urlparse(e["source"]).netloc)
            external_flags.append(int("company" not in e["source"]))

            type_weights.append(e.get("type_weight", 0.5))
            timestamps.append(e.get("timestamp", 0))

        n = len(supports)

        # Evidence metrics
        ESS = self.evidence_metrics.ess(supports, relevances)
        ECS = self.evidence_metrics.ecs(contradictions, relevances)
        EAS = self.evidence_metrics.eas(n)
        ERS = self.evidence_metrics.ers(claim_time, timestamps)
        EStS = self.evidence_metrics.ests(relevances, type_weights)
        EAgS = self.evidence_metrics.eags(supports)

        # Source metrics
        SRS = self.source_metrics.srs(domains)
        EVS = self.source_metrics.evs(external_flags)

        # Claim metrics
        HLS = self.claim_metrics.hls(claim_f)
        CMS = self.claim_metrics.cms(claim_f["entities"])
        CScope = self.claim_metrics.cscope(claim_f["entities"])
        LCS = self.claim_metrics.lcs(claim_f)

        # Aggregation (simple baseline)
        evidence_score = (ESS + ECS + EAS + ERS + EStS + EAgS) / 6
        claim_score = (HLS + CMS + CScope + LCS) / 4

        credibility = self.aggregator.credibility(evidence_score, claim_score, n)

        return {
            "ESS": ESS,
            "ECS": ECS,
            "EAS": EAS,
            "ERS": ERS,
            "EStS": EStS,
            "EAgS": EAgS,
            "SRS": SRS,
            "EVS": EVS,
            "HLS": HLS,
            "CMS": CMS,
            "CScope": CScope,
            "LCS": LCS,
            "Credibility": credibility
        }