class UncertaintyAnalyzer:
    def __init__(self, var_threshold=0.05, high_uncertainty_threshold=0.1):
        self.var_threshold = var_threshold
        self.high_uncertainty_threshold = high_uncertainty_threshold

    def analyze(self, variances):
        """
        variances: dict like {
            "ESS_var": ...,
            "ECS_var": ...,
            "LCS_var": ...,
            ...
        }
        """

        values = list(variances.values())
        mean_var = sum(values) / len(values)

        # identify which metrics are unstable
        high_var_metrics = [
            k for k, v in variances.items()
            if v > self.var_threshold
        ]

        # count instability
        num_high_var = len(high_var_metrics)
        total_metrics = len(variances)

        # normalized disagreement score (0 -> 1)
        disagreement_score = num_high_var / total_metrics

        # high disagreement signal 
        high_disagreement = mean_var > self.var_threshold or num_high_var > 1

        # system-level uncertainty tier
        if mean_var > self.high_uncertainty_threshold:
            uncertainty_level = "high"
        elif mean_var > self.var_threshold:
            uncertainty_level = "medium"
        else:
            uncertainty_level = "low"

        return {
            "mean_variance": mean_var,
            "high_disagreement": high_disagreement,
            "uncertainty_level": uncertainty_level,
            "disagreement_score": disagreement_score,
            "num_unstable_metrics": num_high_var,
            "unstable_metrics": high_var_metrics,
            "confidence": max(0.0, 1 - mean_var)
        }