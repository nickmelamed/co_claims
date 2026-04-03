# TODO determine proper thresholds for escalation 

class EscalationRouter:
    def __init__(
        self,
        var_threshold=0.05,
        conflict_threshold=0.2,
        disagreement_threshold=0.3
    ):
        self.var_threshold = var_threshold
        self.conflict_threshold = conflict_threshold
        self.disagreement_threshold = disagreement_threshold

    def decide(self, metrics, analysis, evidence_count):
        actions = set()

        ESS = metrics["ESS"]
        ECS = metrics["ECS"]

        unstable = set(analysis["unstable_metrics"])
        uncertainty_level = analysis["uncertainty_level"]
        disagreement_score = analysis["disagreement_score"]

        # global uncertainty logic 
        if uncertainty_level == "high":
            # system-wide uncertainty -> strongest response
            actions.update([
                "global_review",
                "more_evidence",
                "debate"
            ])

        elif uncertainty_level == "medium":
            # moderate uncertainty -> selective escalation
            if disagreement_score > self.disagreement_threshold:
                actions.add("debate")

        # metric-specific escalation 

        # evidence issues
        if "ESS_var" in unstable or "ECS_var" in unstable:
            actions.add("more_evidence")

        # logical ambiguity
        if "LCS_var" in unstable:
            actions.add("debate")

        # measurability ambiguity
        if "CMS_var" in unstable:
            actions.add("refine_claim")

        # language ambiguity
        if "HLS_var" in unstable:
            actions.add("rephrase_claim")

        # polarity/semantic conflict
        if abs(ESS - ECS) < self.conflict_threshold:
            actions.add("debate")

        # evidence sufficiency 
        if evidence_count < 2:
            actions.add("more_evidence")

        # clean decision logic 
        if not actions:
            return {
                "decision": "accept",
                "actions": [],
                "confidence": analysis["confidence"],
                "uncertainty": uncertainty_level
            }

        return {
            "decision": "escalate",
            "actions": sorted(actions),
            "confidence": analysis["confidence"],
            "uncertainty": uncertainty_level
        }