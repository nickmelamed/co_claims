#TODO: improve heuristic? 

class EntityResolver:
    def __init__(self, extractor, reasoner=None):
        self.extractor = extractor
        self.reasoner = reasoner

    def should_use_llm(self, claim, base_entities):
        # Heuristic for using llm (basic)
        if len(base_entities) == 0:
            return True

        if len(claim.split()) > 20:
            return True

        trigger_words = ["increase", "decrease", "impact", "affect", "improve"]
        if any(w in claim.lower() for w in trigger_words):
            return True

        return False

    def resolve(self, claim, evidence_list):
        # deterministic extraction
        base = self.extractor.extract(claim)["entities"]

        # llm call decision 
        use_llm = self.reasoner and self.should_use_llm(claim, base)

        structured = None
        llm_entities = set()

        if use_llm:
            structured = self.reasoner.structure(claim)
            llm_entities = set(structured.get("entities", []))

        # filter LLM entities (precision > recall)
        filtered_llm = {
            e for e in llm_entities
            if self._is_valid(e, claim, evidence_list)
        }

        final_entities = set(base).union(filtered_llm)

        return {
            "entities": list(final_entities),
            "structured": structured
        }

    def _is_valid(self, entity, claim, evidence_list):
        entity = entity.lower()

        # appears in claim
        if entity in claim.lower():
            return True

        # appears in evidence
        for e in evidence_list or []:
            if entity in e.get("text", "").lower():
                return True

        return False