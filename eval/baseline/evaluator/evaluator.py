import tldextract

from .extractor import FeatureExtractor
from .similarity import Similarity
from .support import SupportScorer
from .contradiction import ContradictionScorer
from .source_types import classify_source, get_type_weight, is_verifiable
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

            def extract_domain(url):
                ext = tldextract.extract(url)
                return f"{ext.domain}.{ext.suffix}"

            domains.append(extract_domain(e["source"]))

            source_type = e.get("source_type", "unknown")

            source_type = classify_source(e["source"], e["text"])

            type_weight = get_type_weight(source_type)
            external_flag = is_verifiable(source_type)

            type_weights.append(type_weight)
            external_flags.append(external_flag)

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