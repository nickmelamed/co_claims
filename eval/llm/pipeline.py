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
        aggregator
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
        self.aggregator = aggregator

    def run(self, claim):
        # Retrieval 
        evidence_list = self.retriever.retrieve(claim)

        # Embedding 
        claim_embedding = self.embed_fn(claim)

        for e in evidence_list:
            if "embedding" not in e:
                e["embedding"] = self.embed_fn(e["text"])

        # Claim Structuring
        structured = self.reasoner.structure(claim)

        # Relevance Scoring 
        relevances = []
        for e in evidence_list:
            r = e.get("relevance")
            if r is None:
                r = self.triage.sim.relevance(
                    claim_embedding, e["embedding"]
                )
            relevances.append(r)

        # Evidence Triage
        filtered_evidence = self.triage.filter(
            claim_embedding,
            evidence_list
        )

        filtered_relevances = [
            r for e, r in zip(evidence_list, relevances)
            if e in filtered_evidence
        ]

        # LLM Metrics
        metric_outputs = self.metrics.evaluate(
            claim,
            filtered_evidence,
            filtered_relevances
        )

        metrics = metric_outputs['metrics']
        variances = metric_outputs['variances']

        # Uncertainty Analysis
        analysis = self.uncertainty.analyze(variances)

        # Escalation Decision
        decision_obj = self.router.decide(
            metrics,
            analysis,
            len(filtered_evidence)
        )

        # Escalation Action 
        if decision_obj["decision"] == "escalate":

            actions = set(decision_obj["actions"])

            # claim rewrites
            if "rephrase_claim" in actions:
                claim = self.reasoner.rephrase(claim)

            if "refine_claim" in actions:
                claim = self.reasoner.structure(claim).get("refined_claim", claim)

            # global review 
            if "global_review" in actions:
                # full reset -> treat as fresh evaluation
                evidence_list = self.retriever.retrieve(claim, extra=True)

                for e in evidence_list:
                    if "embedding" not in e:
                        e["embedding"] = self.embed_fn(e["text"])

                filtered_evidence = evidence_list

                # recompute everything downstream
                metric_outputs = self.metrics.evaluate(
                                                claim,
                                                filtered_evidence,
                                                filtered_relevances
                                            )
                metrics, variances = metric_outputs['metrics'], metric_outputs['variances']

            else:
                # more evidence
                if "more_evidence" in actions:
                    new_evidence = self.retriever.retrieve(claim, extra=True)

                    for e in new_evidence:
                        if "embedding" not in e:
                            e["embedding"] = self.embed_fn(e["text"])

                    filtered_evidence.extend(new_evidence)

                    # recompute metrics after adding evidence
                    metric_outputs = self.metrics.evaluate(
                                                claim,
                                                filtered_evidence,
                                                filtered_relevances
                                            )
                    metrics, variances = metric_outputs['metrics'], metric_outputs['variances']

            # debate 
            if "debate" in actions:
                debate_output = self.debate_engine.run(claim, filtered_evidence)
                adjudicated = self.adjudicator.decide(claim, debate_output)

                metrics["ESS"] = adjudicated.get("support_score", metrics["ESS"])
                metrics["ECS"] = adjudicated.get("contradiction_score", metrics["ECS"])

                # optionally update variances if debate affects disagreement
                if "variances" in adjudicated:
                    variances.update(adjudicated["variances"])

        # Aggregation 
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
            "variances": variances,
            "decision": decision_obj,
            "credibility": credibility,
            "structured": structured
        }