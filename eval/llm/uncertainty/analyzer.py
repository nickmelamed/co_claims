class UncertaintyAnalyzer:
    def analyze(self, variances):

        variances = list(variances.values())

        return {
            "mean_variance": sum(variances) / len(variances),
            "high_disagreement": any(v > 0.05 for v in variances)
        }