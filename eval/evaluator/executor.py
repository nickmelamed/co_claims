import numpy as np

from eval.evaluator.deterministic.support import SupportScorer
from eval.evaluator.deterministic.contradiction import ContradictionScorer
from eval.evaluator.deterministic.extractor import FeatureExtractor

class UnifiedExecutor:
    def __init__(self, llm_judge, deterministic_metrics, aggregator, use_llm = True):
        self.llm_judge = llm_judge
        self.det = deterministic_metrics
        self.aggregator = aggregator
        self.use_llm = use_llm
        
        self.support = SupportScorer()
        self.contradiction = ContradictionScorer()
        self.extractor = FeatureExtractor()

    async def evaluate(self, claim, claim_time, evidence_list, entities):

        relevances = [e.get("relevance", 0.5) for e in evidence_list]

        if not self.use_llm:

            claim_f = self.extractor.extract(claim)

            evidence_features = [
                self.extractor.extract(e["text"])
                for e in evidence_list
            ]

            supports = [
                self.det.support.score(claim_f, ef)
                for ef in evidence_features
            ]

            contradictions = [
                self.det.contradiction.score(claim_f, ef)
                for ef in evidence_features
            ]

            ess = self.det.ess(supports, relevances)
            ecs = self.det.ecs(contradictions, relevances)
            eags = self.det.eags(supports)
            cms = self.det.cms(claim_f['entities'])
            hls = self.det.hls(claim_f)
            lcs = self.det.lcs(claim_f)

            det_metrics = self.compute_deterministic_metrics(
                claim_time,
                evidence_list,
                per_evidence_scores=[],
                mode='baseline'
            )

            full_metrics = {
                "ESS": ess,
                "ECS": ecs,
                "EAGS": eags,
                "CMS": cms,
                "HLS": hls,
                "LCS": lcs,
                **det_metrics
            }

            claim_score = np.mean([
                cms,
                1 - hls,
                lcs,
                self.det.cscope(claim_f["entities"])
            ])

            evidence_score = np.mean([
                ess,
                1 - ecs,
                det_metrics["EAS"],
                det_metrics["ERS"],
                det_metrics["ESTS"],
                eags,
                det_metrics["SDS"],
                det_metrics["EVS"]
            ])

            final_score = self.aggregator.credibility(
                evidence_score,
                claim_score,
                len(evidence_list)
            )

            return {
                "final_score": final_score,
                "metrics": {
                    **full_metrics,
                    "claim_score": claim_score,
                    "evidence_score": evidence_score
                },
                "variances": {}
            }
        

        llm_metrics, llm_variances, per_evidence_scores = await self.llm_judge.evaluate(
            claim,
            evidence_list,
            relevances
        )

        n = len(evidence_list) if evidence_list else 1

        # claim score
        claim_score, claim_variance = self.compute_claim_score(
            llm_metrics,
            llm_variances,
            entities
        )

        # deterministic scores 
        det_metrics = self.compute_deterministic_metrics(
            claim_time,
            evidence_list,
            per_evidence_scores
        )

        # evidence scores
        evidence_score, evidence_variance = self.compute_evidence_score(
            llm_metrics,
            llm_variances,
            det_metrics
        )

        # penalties
        uncertainty_penalty = np.exp(-evidence_variance)

        evidence_score *= uncertainty_penalty

        # final score
        final_score = self.aggregator.credibility(
            evidence_score,
            claim_score,
            n
        )

        return {
            "final_score": final_score,
            "metrics": {
                **llm_metrics,
                **det_metrics,
                "CScope": self.det.cscope(entities),
                "claim_score": claim_score,
                "evidence_score": evidence_score
            },
            "variances": {
                **llm_variances,
                "claim_variance": claim_variance,
                "evidence_variance": evidence_variance
            }
        }

    # claim score
    def compute_claim_score(self, m, v, entities):
        cscope = self.det.cscope(entities)

        values = [
            m.get("CMS", 0),
            1 - m.get("HLS", 0),  # inverted
            m.get("LCS", 0),
            cscope
        ]

        variances = [
            v.get("CMS", 0),
            v.get("HLS", 0),
            v.get("LCS", 0),
            0.0
        ]

        values = np.array(values, dtype=float)
        variances = np.array(variances, dtype=float)

        claim_score = float(np.mean(values))
        claim_variance = float(np.mean(variances))

        claim_score = self.det._clip(claim_score)

        return claim_score, claim_variance

    # evidence score
    def compute_evidence_score(self, llm_m, llm_v, det_m):
        values = [
            llm_m.get("ESS", 0),
            1 - llm_m.get("ECS", 0),  # should be inverted 
            det_m["EAS"],
            det_m["ERS"],
            det_m["ESTS"],
            det_m["EAGS"],
            det_m["SDS"],
            det_m["EVS"]
        ]

        variances = [
            llm_v.get("ESS", 0),
            llm_v.get("ECS", 0),
            0, 0, 0, 0, 0, 0
        ]

        
        values = np.array(values, dtype=float)
        variances = np.array(variances, dtype=float)

        evidence_score = float(np.mean(values))
        evidence_variance = float(np.mean(variances))

        evidence_score = self.det._clip(evidence_score)

        return evidence_score, evidence_variance

    # deterministic 
    def compute_deterministic_metrics(self, claim_time, evidence_list, per_evidence_scores, mode='full'):
        n = len(evidence_list)

        timestamps = [e.get("timestamp") for e in evidence_list]
        ers = self.det.ers(claim_time, timestamps)

        domains = [
            d if d and d != "" else "unknown"
            for d in [e.get("domain") for e in evidence_list]
        ]
        sds = self.det.sds(domains)

        relevances = [e.get("relevance", 0.5) for e in evidence_list]
        source_types = [e.get("source_type", "unknown") for e in evidence_list]

        # debugging 
        # print("\n=== METRIC INPUT DEBUG ===")
        # print("DOMAINS:", domains)
        # print("TIMESTAMPS:", timestamps)
        # print("SOURCE TYPES:", source_types)

        ests = self.det.ests(relevances, source_types)
        evs = self.det.evs(source_types)

        eas = self.det.eas(n)

        # in baseline eval, don't calculate EAgS
        if mode != 'full':
            return {
            "EAS": eas,
            "ERS": ers,
            "ESTS": ests,
            "SDS": sds,
            "EVS": evs
        }

        # get individual evidence supports 
        supports = [s["ESS"]["score"] for s in per_evidence_scores]
        weights = [s["ESS"]["confidence"] for s in per_evidence_scores]

        eags = self.det.weighted_avg(supports, weights)

        return {
            "EAS": eas,
            "ERS": ers,
            "ESTS": ests,
            "EAGS": eags,
            "SDS": sds,
            "EVS": evs
        }

    def _mean(self, values):
        return sum(values) / len(values) if values else 0.0