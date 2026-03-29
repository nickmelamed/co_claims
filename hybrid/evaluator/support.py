ALPHA_SUPPORT = 0.5

class SupportScorer:
    def overlap(self, T_c, T_e):
        return len(T_c & T_e) / max(1, len(T_c))

    def entity_overlap(self, E_c, E_e):
        return len(E_c & E_e) / max(1, len(E_c))

    def score(self, claim_f, evidence_f):
        return (
            ALPHA_SUPPORT * self.overlap(claim_f["tokens"], evidence_f["tokens"])
            + (1 - ALPHA_SUPPORT) * self.entity_overlap(claim_f["entities"], evidence_f["entities"])
        )