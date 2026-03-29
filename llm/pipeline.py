class EvaluationPipeline:
    def __init__(self, retriever, reasoner, triage, metrics, uncertainty, router):
        self.retriever = retriever
        self.reasoner = reasoner
        self.triage = triage
        self.metrics = metrics
        self.uncertainty = uncertainty
        self.router = router

    def run(self, claim):
        # Retrieval
        evidence = self.retriever.retrieve(claim)

        # Structuring
        structured = self.reasoner.structure(claim)

        # Triage
        filtered = self.triage.filter(...)

        # Metrics
        ess, ecs = self.metrics.score_evidence(claim, filtered)
        cms = self.metrics.score_claim(claim)

        # Uncertainty
        unc = self.uncertainty.analyze(ess + ecs)

        # Escalation
        decision = self.router.decide(unc)

        if decision == "escalate":
            # TODO decide our escalation decision 
            pass

        # Aggregate
        return {
            "metrics": {...},
            "uncertainty": unc,
            "decision": decision
        }