ALPHA_ANTONYM = 0.7
BETA_NEG = 0.3
ENTITY_SOFT_GAMMA = 0.1

NEGATIONS = {"not", "no", "never", "fails"}

ANTONYMS = {
    "increase": "decrease",
    "improve": "worsen",
    "higher": "lower",
    "faster": "slower",
    "more": "less"
}

class ContradictionScorer:

    def entity_soft(self, E_c, E_e):
        overlap = len(E_c & E_e) / max(1, len(E_c))
        return ENTITY_SOFT_GAMMA + (1 - ENTITY_SOFT_GAMMA) * overlap

    def antonym(self, T_c, T_e):
        count = 0
        for w1, w2 in ANTONYMS.items():
            if (w1 in T_c and w2 in T_e) or (w2 in T_c and w1 in T_e):
                count += 1
        return count / max(1, len(T_c))

    def negation(self, T_e):
        return int(any(w in NEGATIONS for w in T_e))

    def overlap(self, T_c, T_e):
        return len(T_c & T_e) / max(1, len(T_c))

    def score(self, claim_f, evidence_f):
        entity_weight = self.entity_soft(claim_f["entities"], evidence_f["entities"])
        antonym_score = self.antonym(claim_f["tokens"], evidence_f["tokens"])
        neg_score = self.negation(evidence_f["tokens"])
        overlap = self.overlap(claim_f["tokens"], evidence_f["tokens"])

        sigma = (ALPHA_ANTONYM * antonym_score + BETA_NEG * neg_score) * overlap
        return entity_weight * sigma