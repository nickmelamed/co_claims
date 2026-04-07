import re

class EntityResolver:
    def __init__(self, extractor, reasoner=None):
        self.extractor = extractor
        self.reasoner = reasoner

    def resolve(self, claim, evidence_list=None):
        evidence_list = evidence_list or []

        # deterministic extraction (fast + precise)
        base_entities = self._normalize_set(
            self.extractor.extract(claim).get("entities", [])
        )

        # decide if LLM is needed
        use_llm = self._should_use_llm(claim, base_entities)

        llm_entities = set()

        if use_llm and self.reasoner:
            raw_llm = self.reasoner.extract_entities(claim)
            llm_entities = self._normalize_set(raw_llm)

        # validate LLM entities (precision > recall)
        filtered_llm = {
            e for e in llm_entities
            if self._is_valid_entity(e, claim, evidence_list)
        }

        # merge
        final_entities = self._dedupe_entities(base_entities.union(filtered_llm))

        return {
            "entities": list(final_entities)
        }

    # LLM trigger 
    def _should_use_llm(self, claim, base_entities):
        words = claim.lower().split()

        # no entities → definitely use LLM
        if len(base_entities) == 0:
            return True

        # vague / abstract claims
        vague_markers = {"impact", "effect", "change", "trend", "growth"}
        if any(w in words for w in vague_markers):
            return True

        # long claims → likely missing entities
        if len(words) > 18:
            return True

        # low coverage (heuristic)
        if len(base_entities) <= 1:
            return True

        return False

    # validation
    def _is_valid_entity(self, entity, claim, evidence_list):
        entity = entity.lower()

        # exact match or word-boundary match in claim
        if self._in_text(entity, claim):
            return True

        # check evidence
        for e in evidence_list:
            if self._in_text(entity, e.get("text", "")):
                return True

        return False

    def _in_text(self, entity, text):
        # avoids substring issues ("ai" matching "said")
        pattern = r"\b{}\b".format(re.escape(entity))
        return re.search(pattern, text.lower()) is not None

    # normalization/dedup
    def _normalize_set(self, entities):
        return {
            self._normalize(e)
            for e in entities
            if isinstance(e, str) and e.strip()
        }

    def _normalize(self, entity):
        entity = entity.lower().strip()

        # remove punctuation noise
        entity = re.sub(r"[^\w\s]", "", entity)

        # collapse whitespace
        entity = re.sub(r"\s+", " ", entity)

        return entity

    def _dedupe_entities(self, entities):
        # simple dedupe now, extensible later
        return set(entities)