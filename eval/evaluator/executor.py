import numpy as np


class UnifiedExecutor:
    def __init__(self, llm_judge, deterministic_metrics, aggregator):
        self.llm_judge = llm_judge
        self.det = deterministic_metrics
        self.aggregator = aggregator

    async def evaluate(self, claim, claim_time, evidence_list, entities):
        # evidence_text = "\n".join([e["text"] for e in evidence_list])

        llm_metrics, llm_variances = self.llm_judge.evaluate(
            claim,
            evidence_list,
            [e.get("relevance", 0.5) for e in evidence_list]
        )

        if llm_metrics is None:
            llm_metrics = {}
        
        if llm_variances is None:
            llm_variances = {}

        # coverage score 
        n = len(evidence_list) if evidence_list else 1

        coverage = {
            "timestamp": sum(e.get("timestamp") is not None for e in evidence_list) / n,
            "domain": sum(e.get("domain") not in [None, "unknown"] for e in evidence_list) / n,
            "source_type": sum(e.get("source_type") != "unknown" for e in evidence_list) / n,
        }

        coverage_score = np.mean(list(coverage.values()))

        # Debugging print statements 
        # print("LLM METRICS:", llm_metrics)
        # print("LLM VARIANCES:", llm_variances)

        # Claim score + variance
        claim_score, claim_variance = self.compute_claim_score(
            llm_metrics,
            llm_variances,
            entities
        )

        # Deterministic metrics (no variance)
        det_metrics = self.compute_deterministic_metrics(
            claim_time,
            evidence_list,
            llm_metrics,
            coverage
        )

        # Evidence score + variance
        evidence_score, evidence_variance = self.compute_evidence_score(
            llm_metrics,
            llm_variances,
            det_metrics
        )

        # Aggregation
        n = len(evidence_list)

        # global coverage penalty 
        evidence_score *= (0.5 + 0.5 * coverage_score)

        final_score = self.aggregator.credibility(
            evidence_score,
            claim_score,
            n
        )

        # Outputs
        all_metrics = {
            **llm_metrics,
            **det_metrics,
            "CScope": self.det.cscope(entities),
            "claim_score": claim_score,
            "evidence_score": evidence_score
        }

        all_variances = {
            **llm_variances,
            "claim_variance": claim_variance,
            "evidence_variance": evidence_variance
        }

        return {
            "final_score": final_score,
            "metrics": all_metrics,
            "variances": all_variances,
            #"raw_judgments": raw
        }

    # claim score
    def compute_claim_score(self, m, v, entities):
        cscope = self.det.cscope(entities)

        values = [
            m.get("CMS", 0),
            m.get("HLS", 0),
            m.get("LCS", 0),
            cscope
        ]

        variances = [
            v.get("CMS", 0),
            v.get("HLS", 0),
            v.get("LCS", 0),
            0.0  # deterministic metric
        ]

        return self._mean(values), self._mean(variances)

    # evidence scores
    def compute_evidence_score(self, llm_m, llm_v, det_m):
        values = [
            llm_m.get("ESS", 0),
            1 - llm_m.get("ECS", 0),
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
            0.0,  # deterministic
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        return self._mean(values), self._mean(variances)

    # deterministic metrics
    def compute_deterministic_metrics(self, claim_time, evidence_list, llm_metrics, coverage):
        n = len(evidence_list)

        timestamps = [e.get("timestamp", claim_time) for e in evidence_list]

        valid_times = [t for t in timestamps if t is not None]

        if not valid_times:
            ers = 0.5 # fallback if no times
        else:
            ers = self.det.ers(claim_time, valid_times)
        
        ers *= coverage['timestamp']

        domains = [e.get("domain", "unknown") for e in evidence_list]

        srs = self.det.srs(domains) * coverage['domain']

        relevances = [e.get("relevance", 0.5) for e in evidence_list]
        source_types = [e.get("source_type", "unknown") for e in evidence_list]

        ests = self.det.ests(relevances, source_types) * coverage['source_type']
        evs = self.det.evs(source_types) * coverage['source_type']

        supports = [llm_metrics.get("ESS", 0.5)] * n
        eags = self.det.eags(supports)

        
        

        return {
            "EAS": self.det.eas(n),
            "ERS": ers,
            "ESTS": ests,
            "EAGS": eags,
            "SRS": srs,
            "EVS": evs
        }
    


    # utils
    def _mean(self, values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _clip(self, x):
        return max(0.0, min(1.0, x))