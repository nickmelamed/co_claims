import numpy as np

class JudgeEnsemble:
    def __init__(self, judges):
        self.judges = judges

    def evaluate(self, prompt):
        outputs = [j.evaluate(prompt) for j in self.judges]

        # filter valid outputs
        valid = [o for o in outputs if isinstance(o, dict)]

        if not valid:
            return {}, {}, outputs

        metrics = valid[0].keys()

        aggregated_scores = {}
        aggregated_variances = {}

        for m in metrics:
            values = []
            weights = []

            for o in valid:
                if m in o:
                    val = o[m]

                    if isinstance(val, dict) and "score" in val:
                        values.append(val["score"])
                        weights.append(val.get("confidence", 1.0))

                    elif isinstance(val, (int, float)):
                        values.append(val)
                        weights.append(1.0)

            if not values:
                aggregated_scores[m] = 0.0
                aggregated_variances[m] = 1.0
                continue

            mean = np.average(values, weights=weights)
            variance = np.var(values)

            aggregated_scores[m] = float(mean)
            aggregated_variances[m] = float(variance)

        return aggregated_scores, aggregated_variances, valid