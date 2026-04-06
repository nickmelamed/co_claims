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
        debate_engine,
        adjudicator,
    ):
        self.retriever = retriever
        self.embed_fn = embed_fn
        self.reasoner = reasoner
        self.triage = triage
        self.metrics = metric_executor
        self.uncertainty = uncertainty_analyzer
        self.router = escalation_router
        self.debate_engine = debate_engine
        self.adjudicator = adjudicator

    def _embed_evidence(self, evidence_list):
        for e in evidence_list:
            if "embedding" not in e:
                e["embedding"] = self.embed_fn(e["text"])

    def _attach_relevance(self, claim_embedding, evidence_list):
        for e in evidence_list:
            if "relevance" not in e:
                e["relevance"] = self.triage.sim.relevance(
                    claim_embedding, e["embedding"]
                )

    def _evaluate(self, claim, structured, evidence_list):
        return self.metrics.evaluate(
            claim=claim,
            claim_time=structured.get("claim_time"),
            evidence_list=evidence_list,
            entities=structured.get("entities", [])
        )

    def run(self, claim):
        # retrieval 
        evidence_list = self.retriever.retrieve(claim)

        # embedding 
        claim_embedding = self.embed_fn(claim)
        self._embed_evidence(evidence_list)

        # claim structuring 
        structured = self.reasoner.structure(claim)

        # relevance and triage 
        self._attach_relevance(claim_embedding, evidence_list)

        filtered_evidence = self.triage.filter(
            claim_embedding,
            evidence_list
        )

        # initial evaluation 
        metric_outputs = self._evaluate(
            claim,
            structured,
            filtered_evidence
        )

        metrics = metric_outputs["metrics"]
        variances = metric_outputs["variances"]

        # uncertainty analysis 
        analysis = self.uncertainty.analyze(variances)

        # escalation decision 
        decision_obj = self.router.decide(
            metrics,
            analysis,
            len(filtered_evidence)
        )

        # escalation actions 
        if decision_obj["decision"] == "escalate":

            actions = set(decision_obj["actions"])

            # Claim modifications
            if "rephrase_claim" in actions:
                claim = self.reasoner.rephrase(claim)

            if "refine_claim" in actions:
                structured = self.reasoner.structure(claim)
                claim = structured.get("refined_claim", claim)

            # Global reset
            if "global_review" in actions:
                evidence_list = self.retriever.retrieve(claim, extra=True)
                self._embed_evidence(evidence_list)
                self._attach_relevance(claim_embedding, evidence_list)

                filtered_evidence = evidence_list

            else:
                if "more_evidence" in actions:
                    new_evidence = self.retriever.retrieve(claim, extra=True)

                    self._embed_evidence(new_evidence)
                    self._attach_relevance(claim_embedding, new_evidence)

                    filtered_evidence.extend(new_evidence)

            # Re-evaluate after changes
            metric_outputs = self._evaluate(
                claim,
                structured,
                filtered_evidence
            )

            metrics = metric_outputs["metrics"]
            variances = metric_outputs["variances"]

            # Debate (post-metrics override)
            if "debate" in actions:
                debate_output = self.debate_engine.run(claim, filtered_evidence)
                adjudicated = self.adjudicator.decide(claim, debate_output)

                metrics["ESS"] = adjudicated.get("support_score", metrics["ESS"])
                metrics["ECS"] = adjudicated.get("contradiction_score", metrics["ECS"])

                if "variances" in adjudicated:
                    variances.update(adjudicated["variances"])

        # final output 
        return {
            "metrics": metrics,
            "variances": variances,
            "decision": decision_obj,
            "credibility": metric_outputs["final_score"],
            "structured": structured,
            "raw_judgments": metric_outputs.get("raw_judgments")
        }