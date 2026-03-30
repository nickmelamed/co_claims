class UncertaintyAnalyzer:
    def analyze(self, metric_outputs):
        variances = [m["variance"] for m in metric_outputs]

        return {
            "mean_variance": sum(variances) / len(variances),
            "high_disagreement": any(v > 0.05 for v in variances)
        }