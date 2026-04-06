import numpy as np
import asyncio

class JudgeEnsemble:
    def __init__(self, judges):
        self.judges = judges

    async def _eval_one(self, judge, prompt):
        return await asyncio.to_thread(judge.evaluate, prompt)

    async def evaluate_async(self, prompt):
        tasks = [
            self._eval_one(j, prompt)
            for j in self.judges
        ]

        outputs = await asyncio.gather(*tasks)

        return self._aggregate(outputs)

    def evaluate(self, prompt):
        """
        Backward-compatible sync version
        """
        outputs = [j.evaluate(prompt) for j in self.judges]
        return self._aggregate(outputs)

    def _aggregate(self, outputs):
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