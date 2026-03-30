# TODO determine proper variance threshold for escalation 

class EscalationRouter:
    def __init__(self, var_threshold=0.05, conflict_threshold=0.2):
        self.var_threshold = var_threshold
        self.conflict_threshold = conflict_threshold

    def decide(self, metrics, uncertainty, evidence_count):
        ESS = metrics["ESS"]
        ECS = metrics["ECS"]

        # 1. Global disagreement
        if uncertainty["mean_variance"] > self.var_threshold:
            return "escalate"

        # 2. Polarity conflict
        if abs(ESS - ECS) < self.conflict_threshold:
            return "escalate"

        # 3. Logical ambiguity
        if uncertainty["LCS_var"] > self.var_threshold:
            return "escalate"

        # 4. Measurability ambiguity
        if uncertainty["CMS_var"] > self.var_threshold:
            return "escalate"

        # 5. Low evidence
        if evidence_count < 2:
            return "escalate"

        return "accept"