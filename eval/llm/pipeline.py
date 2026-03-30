class EvaluationPipeline:
    def __init__(
        self,
        retriever,
        embed_fn,
        reasoner,
        triage,
        metric_executor,
        uncertainty_analyzer,
        escalation_router,
        aggregator
    ):
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.reasoner = reasoner
        self.triage = triage
        self.metrics = metric_executor
        self.uncertainty = uncertainty_analyzer
        self.router = escalation_router
        self.aggregator = aggregator

    def run(self, claim):
        # retrieval 
        evidence_list = self.retriever.retrieve(claim)

        # structuring (LLM-reasoning)
        structured = self.reasoner.structure(claim)

        # relevance score
        claim_embedding = self.embed_fn(claim)

        relevances = []
        for e in evidence_list:
            r = e.get("relevance")
            if r is None:
                r = self.triage.sim.relevance(
                    claim_embedding, e["embedding"]
                )
            relevances.append(r)

        # evidence triage 
        filtered_evidence = self.triage.filter(
            claim_embedding,
            evidence_list
        )

        filtered_relevances = [
            r for e, r in zip(evidence_list, relevances)
            if e in filtered_evidence
        ]

        # LLM metrics 
        metric_outputs = self.metrics.evaluate(
            claim,
            filtered_evidence,
            filtered_relevances
        )

        metrics = metric_outputs['metrics']
        uncertainty = metric_outputs['uncertainty']

        # uncertainty analysis 
        uncertainty = self.uncertainty.analyze(metric_outputs)


        # adaptive escalation decision 
        decision_obj = self.router.decide(
            metrics,
            uncertainty,
            len(filtered_evidence)
        )

        # escalation actions 
        if decision_obj["decision"] == "escalate":

            if "more_evidence" in decision_obj["actions"]:
                new_evidence = self.retriever.retrieve(claim, extra=True)
                filtered_evidence.extend(new_evidence)

            if "refine_claim" in decision_obj["actions"]:
                claim = self.reasoner.structure(claim).get("refined_claim", claim)

            if "debate" in decision_obj["actions"]:
                debate_output = self.debate_engine.run(claim, filtered_evidence)
                adjudicated = self.adjudicator.decide(claim, debate_output)

                metrics["ESS"] = adjudicated.get("support_score", metrics["ESS"])
                metrics["ECS"] = adjudicated.get("contradiction_score", metrics["ECS"])

        # aggregation 
        n = len(filtered_evidence)

        evidence_score = (metrics["ESS"] + metrics["ECS"]) / 2
        claim_score = (metrics["CMS"] + metrics["LCS"] + metrics["HLS"]) / 3

        credibility = self.aggregator.credibility(
            evidence_score,
            claim_score,
            n
        )

        return {
            "metrics": metrics,
            "uncertainty": uncertainty,
            "decision": decision_obj,
            "credibility": credibility,
            "structured": structured
        }