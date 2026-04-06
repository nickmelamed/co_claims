class EvaluationPipeline:
    def __init__(
        self,
        embed_fn,
        entity_resolver,
        reasoner,
        triage,
        metric_executor,
        uncertainty_analyzer,
        escalation_router,
        debate_engine,
        adjudicator,
    ):
        self.embed_fn = embed_fn
        self.entity_resolver = entity_resolver
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

    def _evaluate(self, claim, structured, evidence_list, entities):
        return self.metrics.evaluate(
            claim=claim,
            claim_time=structured.get("claim_time"),
            evidence_list=evidence_list,
            entities=entities,
        )

    def run(self, claim, evidence_list):

        # embedding 
        claim_embedding = self.embed_fn(claim)
        self._embed_evidence(evidence_list)

        # claim structuring w/ entity resolution 

        resolved = self.entity_resolver(
            claim,
            evidence_list
        )

        entities = resolved['entities']
        structured = resolved['structured']

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
            filtered_evidence,
            entities
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
            if "rephrase_claim" in actions and self.entity_resolve.reasoner:
                claim = self.entity_resolver.reasoner.rephrase(claim)

            if "refine_claim" in actions and self.entity_resolver.reasoner:
                structured = self.entity_resolver.reasoner.structure(claim)
                claim = structured.get("refined_claim", claim)

            # Global reset
            # TODO: implement proper logic given new RAG 
            if "global_review" in actions:
                # evidence_list = self.retriever.retrieve(claim, extra=True)
                # self._embed_evidence(evidence_list)
                # self._attach_relevance(claim_embedding, evidence_list)

                # filtered_evidence = evidence_list
                pass

            else:
                # if "more_evidence" in actions:
                #     new_evidence = self.retriever.retrieve(claim, extra=True)

                #     self._embed_evidence(new_evidence)
                #     self._attach_relevance(claim_embedding, new_evidence)

                #     filtered_evidence.extend(new_evidence)
                pass

            # Re-evaluate after changes

            resolved = self.entity_resolver.resolve(claim, filtered_evidence)
            entities = resolved['metrics']
            structured = resolved['structured']

            metric_outputs = self._evaluate(
                claim,
                structured,
                filtered_evidence,
                entities
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
            "entities": entities,
            "raw_judgments": metric_outputs.get("raw_judgments")
        }