import numpy as np

class UnifiedExecutor:
    def __init__(self, llm_judge, deterministic_metrics, aggregator):
        self.llm_judge = llm_judge
        self.det = deterministic_metrics
        self.aggregator = aggregator

    async def evaluate(self, claim, claim_time, evidence_list, entities):

        relevances = [e.get("relevance", 0.5) for e in evidence_list]

        llm_metrics, llm_variances, per_evidence_scores = await self.llm_judge.evaluate(
            claim,
            evidence_list,
            relevances
        )

        n = len(evidence_list) if evidence_list else 1

        # coverage
        coverage = {
            "timestamp": sum(e.get("timestamp") is not None for e in evidence_list) / n,
            "domain": sum(e.get("domain") not in [None, "unknown"] for e in evidence_list) / n,
            "source_type": sum(e.get("source_type") != "unknown" for e in evidence_list) / n,
        }

        coverage_score = np.mean(list(coverage.values()))

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
            coverage,
            per_evidence_scores
        )

        # evidence scores
        evidence_score, evidence_variance = self.compute_evidence_score(
            llm_metrics,
            llm_variances,
            det_metrics
        )

        # penalties
        coverage_penalty = (0.5 + 0.5 * coverage_score)
        uncertainty_penalty = 1 - min(1, evidence_variance)

        evidence_score *= coverage_penalty * uncertainty_penalty

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
            1 - m.get("HLS", 0),  # ✅ FIX
            m.get("LCS", 0),
            cscope
        ]

        variances = [
            v.get("CMS", 0),
            v.get("HLS", 0),
            v.get("LCS", 0),
            0.0
        ]

        return self._mean(values), self._mean(variances)

    # evidence score
    def compute_evidence_score(self, llm_m, llm_v, det_m):
        values = [
            llm_m.get("ESS", 0),
            1 - llm_m.get("ECS", 0),  # should be inverted 
            det_m["EAS"],
            det_m["ERS"],
            det_m["ESTS"],
            det_m["EAGS"],
            det_m["SRS"],
            det_m["EVS"]
        ]

        variances = [
            llm_v.get("ESS", 0),
            llm_v.get("ECS", 0),
            0, 0, 0, 0, 0, 0
        ]

        return self._mean(values), self._mean(variances)

    # deterministic 
    def compute_deterministic_metrics(self, claim_time, evidence_list, coverage, per_evidence_scores):
        n = len(evidence_list)

        timestamps = [e.get("timestamp") for e in evidence_list if e.get("timestamp") is not None]

        if not timestamps:
            ers = 0.5
        else:
            ers = self.det.ers(claim_time, timestamps)

        ers *= coverage['timestamp']

        domains = [e.get("domain", "unknown") for e in evidence_list]
        srs = self.det.srs(domains) * coverage['domain']

        relevances = [e.get("relevance", 0.5) for e in evidence_list]
        source_types = [e.get("source_type", "unknown") for e in evidence_list]

        ests = self.det.ests(relevances, source_types) * coverage['source_type']
        evs = self.det.evs(source_types) * coverage['source_type']

        # get individual evidence supports 
        supports = [s.get("ESS", 0.5) for s in per_evidence_scores]
        eags = self.det.eags(supports)

        return {
            "EAS": self.det.eas(n),
            "ERS": ers,
            "ESTS": ests,
            "EAGS": eags,
            "SRS": srs,
            "EVS": evs
        }

    def _mean(self, values):
        return sum(values) / len(values) if values else 0.0