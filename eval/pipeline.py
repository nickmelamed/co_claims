import asyncio
import numpy as np

class EvaluationPipeline:
    def __init__(
        self,
        embed_fn,
        embed_batch_fn,
        entity_resolver,
        reasoner,
        triage,
        metric_executor,
        uncertainty_analyzer,
        escalation_router,
        debate_engine,
        adjudicator,
        retrieve_fn=None,
        mode='full',
    ):
        self.embed_fn = embed_fn
        self.embed_batch_fn = embed_batch_fn
        self.entity_resolver = entity_resolver
        self.reasoner = reasoner
        self.triage = triage
        self.metrics = metric_executor
        self.uncertainty = uncertainty_analyzer
        self.router = escalation_router
        self.debate_engine = debate_engine
        self.adjudicator = adjudicator
        self.retrieve_fn = retrieve_fn
        self.mode = mode

    def _embed_evidence(self, evidence_list):
        missing = [e for e in evidence_list if "embedding" not in e]

        if not missing:
            return

        texts = [e["text"] for e in missing]

        embeddings = self.embed_batch_fn(texts)

        for e, emb in zip(missing, embeddings):
            e["embedding"] = emb

    def _attach_relevance(self, claim_embedding, evidence_list):
        for e in evidence_list:
            if "relevance" not in e:
                e["relevance"] = self.triage.sim.relevance(
                    claim_embedding, e["embedding"]
                )

    async def _evaluate(self, claim, claim_time, evidence_list, entities):
        return await self.metrics.evaluate(
            claim=claim,
            claim_time=claim_time,
            evidence_list=evidence_list,
            entities=entities,
        )

    async def run(self, claim, evidence_list):

        # run embedding and resolved in parallel 

        claim_embedding_task = asyncio.to_thread(self.embed_fn, claim)
        resolved_task = asyncio.to_thread(self.entity_resolver.resolve, claim, evidence_list)

        claim_embedding, resolved = await asyncio.gather(
            claim_embedding_task,
            resolved_task
        )

        if resolved is None:
            resolved = {"entities": []}

        entities = resolved.get("entities") or []

        # debugging 
        if entities is None:
            entities = []
            print("Reasoner returned no entities for: ", claim)

        claim_time = self.reasoner.extract_time(claim)

        # embedding 
        self._embed_evidence(evidence_list)

        # relevance and triage 
        self._attach_relevance(claim_embedding, evidence_list)

        filtered_evidence = self.triage.filter(
            claim_embedding,
            evidence_list
        )

        # # debugging 
        # print("EVIDENCE BEFORE TRIAGE:", len(evidence_list))
        # print("EVIDENCE AFTER TRIAGE:", len(filtered_evidence))


        metric_outputs = await self._evaluate(
            claim,
            claim_time,
            filtered_evidence,
            entities
        )

        metrics = metric_outputs["metrics"]
        variances = metric_outputs["variances"]

        # skipping escalation (if single)

        if self.mode != 'full':
            return {
                "metrics": metrics,
                "variances": variances,
                "credibility": metric_outputs["final_score"],
                "entities": entities,
                "mode": self.mode
            }
        
        # escalation (for full mode only)

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
            if "rephrase_claim" in actions and self.entity_resolver.reasoner:
                claim = await asyncio.to_thread(
                    self.entity_resolver.reasoner.rephrase,
                    claim
                )

                claim_embedding = await asyncio.to_thread(self.embed_fn, claim)

            if self.retrieve_fn:

                # GLOBAL RESET (strongest action)
                if "global_review" in actions:
                    new_evidence = await self.retrieve_fn(claim, extra=True)

                    self._embed_evidence(new_evidence)
                    self._attach_relevance(claim_embedding, new_evidence)

                    filtered_evidence = self.triage.filter(
                        claim_embedding,
                        new_evidence
                    )

                # INCREMENTAL RETRIEVAL
                elif "more_evidence" in actions:
                    new_evidence = await self.retrieve_fn(claim, extra=True)

                    # deduplicate by text or url
                    existing_texts = set(e["text"] for e in filtered_evidence)

                    new_evidence = [
                        e for e in new_evidence
                        if e["text"] not in existing_texts
                    ]

                    if new_evidence:
                        self._embed_evidence(new_evidence)
                        self._attach_relevance(claim_embedding, new_evidence)

                        filtered_evidence.extend(new_evidence)

                        # re-triage after adding
                        filtered_evidence = self.triage.filter(
                            claim_embedding,
                            filtered_evidence
                        )

            # Re-evaluate after changes

            resolved = await asyncio.to_thread(
                self.entity_resolver.resolve,
                claim,
                filtered_evidence
            )

            if resolved is None: 
                resolved = {"entities": []}

            entities = (resolved or {}).get("entities") or []
        
            if entities is None:
                entities = []
                print("Reasoner returned no entities for: ", claim)

            claim_time = self.reasoner.extract_time(claim)

            metric_outputs = await self._evaluate(
                claim,
                claim_time,
                filtered_evidence,
                entities
            )

            metrics = metric_outputs["metrics"]
            variances = metric_outputs["variances"]

            # Debate (post-metrics override)
            if "debate" in actions:
                debate_output = await self.debate_engine.run_async(claim, filtered_evidence)
                adjudicated = await asyncio.to_thread(
                    self.adjudicator.decide,
                    claim,
                    debate_output
                ) or {}

                metrics["ESS"] = adjudicated.get("support_score", metrics.get("ESS", 0))
                metrics["ECS"] = adjudicated.get("contradiction_score", metrics.get("ECS", 0))

                if "variances" in adjudicated:
                    variances.update(adjudicated["variances"])

        # final output 
        return {
            "metrics": metrics,
            "variances": variances,
            "decision": decision_obj,
            "credibility": metric_outputs["final_score"],
            "entities": entities,
            "mode": "full"
        }