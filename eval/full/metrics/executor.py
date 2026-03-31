import numpy as np


class UnifiedExecutor:
    def __init__(self, ensemble, deterministic_metrics):
        self.ensemble = ensemble
        self.det = deterministic_metrics

    def evaluate(self, claim, claim_f, evidence_list, domains):
        evidence_text = "\n".join([e["text"] for e in evidence_list])

        # LLM metrics 
        llm_metrics, variance, raw = self.ensemble.evaluate(
            claim,
            evidence_text
        )

        # Deterministic metrics 
        evidence_entities = [e["entities"] for e in evidence_list]
        n = len(evidence_list)

        det_metrics = self.det.compute(
            claim_f,
            evidence_entities,
            domains,
            n
        )

        # merge metrics
        all_metrics = {**llm_metrics, **det_metrics}

        # uncertainty calculation
        mean_variance = np.mean(list(variance.values()))

        uncertainty = {
            **{f"{k}_var": v for k, v in variance.items()},
            "mean_variance": mean_variance
        }

        return {
            "metrics": all_metrics,
            "uncertainty": uncertainty,
            "raw_judgments": raw
        }