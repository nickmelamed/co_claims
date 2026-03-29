import numpy as np

class JudgeEnsemble:
    def __init__(self, judges):
        self.judges = judges

    def evaluate(self, prompt):
        outputs = [j.evaluate(prompt) for j in self.judges]

        valid = [o for o in outputs if "error" not in o]

        entailments = [o["entailment"] for o in valid]
        contradictions = [o["contradiction"] for o in valid]

        return {
            "entailment_mean": np.mean(entailments),
            "contradiction_mean": np.mean(contradictions),
            "variance": np.var(entailments + contradictions),
            "raw": valid
        }