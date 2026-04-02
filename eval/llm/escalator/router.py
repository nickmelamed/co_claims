# TODO determine proper thresholds for escalation 

class EscalationRouter:
    def __init__(self, var_threshold=0.05, conflict_threshold=0.2):
        self.var_threshold = var_threshold
        self.conflict_threshold = conflict_threshold

    def decide(self, metrics, variances, evidence_count):
        actions = []

        ESS = metrics["ESS"]
        ECS = metrics["ECS"]

        # evidence issues
        if variances["ESS_var"] > self.var_threshold or \
           variances["ECS_var"] > self.var_threshold:
            actions.append("more_evidence")

        # polarity conflict 
        if abs(ESS - ECS) < self.conflict_threshold:
            actions.append("debate")

        # logical ambiguity 
        if variances["LCS_var"] > self.var_threshold:
            actions.append("debate")

        # measurability ambiguity 
        if variances["CMS_var"] > self.var_threshold:
            actions.append("refine_claim")

        # language ambiguity 
        if variances["HLS_var"] > self.var_threshold:
            actions.append("rephrase_claim")

        # evidence shortage 
        if evidence_count < 2:
            actions.append("more_evidence")

        if not actions:
            return {"decision": "accept", "actions": []}

        return {"decision": "escalate", "actions": list(set(actions))}