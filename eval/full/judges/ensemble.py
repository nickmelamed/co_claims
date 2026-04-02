import numpy as np


class JudgeEnsemble:
    def __init__(self, judges):
        self.judges = judges

    def evaluate(self, prompt, field="score"):
        """
        field: which key to aggregate ("score", "entailment", etc.)
        """

        outputs = [j.evaluate(prompt) for j in self.judges]

        valid = [o for o in outputs if isinstance(o, dict) and field in o]

        if not valid:
            return {
                "mean": 0.0,
                "variance": 1.0,
                "raw": outputs
            }

        values = [o[field] for o in valid]

        # use confidence-weighted averaging
        if "confidence" in valid[0]:
            weights = [o.get("confidence", 1.0) for o in valid]
            mean = np.average(values, weights=weights)
        else:
            mean = np.mean(values)

        variance = np.var(values)

        return {
            "mean": float(mean),
            "variance": float(variance),
            "raw": valid
        }