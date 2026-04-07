import numpy as np
import asyncio

METRICS = ["ESS", "ECS", "CMS", "LCS", "HLS"]

DEFAULT_METRIC = {
    "score": 0.0,
    "confidence": 0.0
}

DEFAULT_SCHEMA = {
    m: DEFAULT_METRIC.copy() for m in METRICS
}

class JudgeEnsemble:
    def __init__(self, judges):
        self.judges = judges

    def _apply_schema(self, output):
        if not isinstance(output, dict):
            output = {}

        structured = {}

        for m in METRICS:
            val = output.get(m, {})

            if not isinstance(val, dict):
                val = {}

            structured[m] = {
                "score": val.get("score", 0.0),
                "confidence": val.get("confidence", 0.0)
            }

        return structured
    
    def _normalize(self, structured):
        for m in METRICS:
            s = structured[m]["score"]
            c = structured[m]["confidence"]

            try:
                s = float(s)
            except:
                s = 0.0

            try:
                c = float(c)
            except:
                c = 0.0

            structured[m]["score"] = max(0.0, min(1.0, s))
            structured[m]["confidence"] = max(0.0, min(1.0, c))

        return structured

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
        structured_outputs = []

        for o in outputs:
            try:
                s = self._apply_schema(o)
                s = self._normalize(s)
                structured_outputs.append(s)
            except Exception:
                continue

        if not structured_outputs:
            return (
                {m: 0.0 for m in METRICS},
                {m: 1.0 for m in METRICS},
                outputs
            )
        
        aggregated_scores = {}
        aggregated_variances = {}

        for m in METRICS:
            values = []
            weights = []

            for o in structured_outputs:
                values.append(o[m]["score"])
                weights.append(o[m]["confidence"])

            if not values:
                aggregated_scores[m] = 0.0
                aggregated_variances[m] = 1.0
                continue

            mean = np.average(values, weights=weights)
            variance = np.average((values - mean) ** 2, weights=weights)

            aggregated_scores[m] = float(mean)
            aggregated_variances[m] = float(variance)

        return aggregated_scores, aggregated_variances, structured_outputs