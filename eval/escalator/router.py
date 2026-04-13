class EscalationRouter:
    def __init__(
        self,
        var_threshold=0.03,
        conflict_threshold=0.15,
        disagreement_threshold=0.25,
        uncertainty_threshold=0.5,
        soft_threshold=0.3
    ):
        self.var_threshold = var_threshold
        self.conflict_threshold = conflict_threshold
        self.disagreement_threshold = disagreement_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.soft_threshold = soft_threshold

    def _compute_uncertainty_score(self, metrics, analysis, evidence_count):
        score = 0.0

        ESS = metrics["ESS"]
        ECS = metrics["ECS"]

        unstable = set(analysis["unstable_metrics"])
        disagreement = analysis["disagreement_score"]
        uncertainty_level = analysis["uncertainty_level"]

        # variance/instability contribution 
        # each unstable metric contributes signal (capped)
        var_signal = min(len(unstable) * 0.12, 0.36)
        score += var_signal

        # disagreement contribution 
        score += min(disagreement, 0.3)

        # margin-based uncertainty as conflict 
        margin = abs(ESS - ECS)
        if margin < self.conflict_threshold:
            # stronger penalty the closer to zero margin
            score += (self.conflict_threshold - margin) / self.conflict_threshold * 0.25

        # evidence sufficiency 
        if evidence_count < 2:
            score += 0.25
        elif evidence_count < 4:
            score += 0.1

        # global uncertainty level 
        if uncertainty_level == "high":
            score += 0.3
        elif uncertainty_level == "medium":
            score += 0.15

        return min(score, 1.0), margin, unstable

    def decide(self, metrics, analysis, evidence_count):
        actions = set()

        score, margin, unstable = self._compute_uncertainty_score(
            metrics, analysis, evidence_count
        )

        disagreement = analysis["disagreement_score"]
        uncertainty_level = analysis["uncertainty_level"]

        # decision 
        if score > self.uncertainty_threshold:
            decision = "escalate"
        elif score > self.soft_threshold:
            decision = "escalate"  # soft escalation still triggers actions
        else:
            return {
                "decision": "accept",
                "actions": [],
                "confidence": analysis["confidence"],
                "uncertainty": uncertainty_level
            }

        # action selection 

        # Evidence issues
        if "ESS_var" in unstable or "ECS_var" in unstable or evidence_count < 2:
            actions.update(["more_evidence"])

        # Logical ambiguity -> debate
        if "LCS_var" in unstable or margin < 0.1:
            actions.update(["debate"])

        # Disagreement-driven debate
        if disagreement > self.disagreement_threshold:
            actions.update(["debate"])

        # Claim refinement
        if "CMS_var" in unstable and evidence_count > 2:
            actions.update(["refine_claim"])

        # Language ambiguity
        if "HLS_var" in unstable:
            actions.update(["rephrase_claim"])

        # High uncertainty → strongest response
        if uncertainty_level == "high" or score > 0.7:
            actions.update(["global_review", "more_evidence", "debate"])

        return {
            "decision": decision,
            "actions": sorted(actions),
            "confidence": analysis["confidence"],
            "uncertainty": uncertainty_level,
            "uncertainty_score": score  # useful for debugging/logging
        }