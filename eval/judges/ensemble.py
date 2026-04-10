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
            return DEFAULT_SCHEMA.copy()

        structured = {}

        for m in METRICS:
            val = output.get(m, {})

            # CASE 1: already correct
            if isinstance(val, dict):
                score = val.get("score", 0.0)
                conf = val.get("confidence", 0.0)

            # CASE 2: raw float score
            elif isinstance(val, (int, float)):
                score = val
                conf = 0.5  # fallback confidence

            # CASE 3: missing
            else:
                score = 0.0
                conf = 0.0

            structured[m] = {
                "score": score,
                "confidence": conf
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

        # debugging 

        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        clean_outputs = []
        for o in outputs:
            if isinstance(o, Exception):
                print("JUDGE FAILED:", o, flush=True)
                continue
            clean_outputs.append(o)

        # debugging 
        print("=== ENSEMBLE RAW OUTPUTS ===")
        for i, o in enumerate(outputs):
            print(f"Judge {i}: type={type(o)} value={o}")

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
                weights.append(o[m]["confidence"] + 1e-3)
                print("RAW JUDGE OUTPUT:", type(o), o) # debugging

            if not values:
                aggregated_scores[m] = 0.0
                aggregated_variances[m] = 1.0
                continue

            mean = np.average(values, weights=weights)
            variance = np.average((values - mean) ** 2, weights=weights)

            aggregated_scores[m] = float(mean)
            aggregated_variances[m] = float(variance)

        return aggregated_scores, aggregated_variances, structured_outputs