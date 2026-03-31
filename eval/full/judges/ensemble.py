import numpy as np


import numpy as np


class JudgeEnsemble:
    def __init__(self, prometheus, mixtral):
        self.prometheus = prometheus
        self.mixtral = mixtral

    def evaluate(self, claim, evidence_text):
        p = self.prometheus.evaluate(claim, evidence_text)
        m = self.mixtral.evaluate(claim, evidence_text)

        outputs = [p, m]

        metrics = ["ESS", "ECS", "CMS", "LCS", "HLS"]

        result = {}
        variance = {}

        for metric in metrics:
            vals = [o[metric] for o in outputs if metric in o]

            if not vals:
                result[metric] = 0
                variance[metric] = 1
                continue

            result[metric] = float(np.mean(vals))
            variance[metric] = float(np.var(vals))

        return result, variance, outputs