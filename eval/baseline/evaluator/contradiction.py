ALPHA_ANTONYM = 0.7
BETA_NEG = 0.3
ENTITY_SOFT_GAMMA = 0.1

NEGATIONS = {
    "not", "no", "never", "none", "nothing", "nowhere",
    "neither", "nor", "cannot", "can't", "won't", "shouldn't",
    "wouldn't", "couldn't", "doesn't", "don't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "without",
    "lack", "lacks", "lacking", "lacked", "fails", "failed",
    "failing", "fail", "absence", "missing"
}   

ANTONYMS = {
    # Direction / Change
    "increase": {"decrease", "decline", "drop", "reduce"},
    "decrease": {"increase", "rise", "grow"},
    "rise": {"fall", "drop", "decline"},
    "fall": {"rise", "increase", "grow"},
    "grow": {"shrink", "decline", "contract"},
    "shrink": {"grow", "expand"},

    # Quality / Performance
    "improve": {"worsen", "decline", "deteriorate"},
    "worsen": {"improve", "enhance"},
    "better": {"worse"},
    "worse": {"better"},
    "efficient": {"inefficient"},
    "inefficient": {"efficient"},

    # Quantity / Magnitude
    "more": {"less", "fewer"},
    "less": {"more"},
    "greater": {"smaller", "lesser"},
    "smaller": {"larger", "greater"},
    "large": {"small"},
    "small": {"large"},
    "high": {"low"},
    "low": {"high"},
    "maximum": {"minimum"},
    "minimum": {"maximum"},

    # Speed / Time
    "fast": {"slow"},
    "faster": {"slower"},
    "slow": {"fast"},
    "early": {"late"},
    "late": {"early"},
    "before": {"after"},
    "after": {"before"},

    # Polarity / Sentiment
    "positive": {"negative"},
    "negative": {"positive"},
    "good": {"bad"},
    "bad": {"good"},
    "beneficial": {"harmful"},
    "harmful": {"beneficial"},

    # Existence / Presence
    "present": {"absent", "missing"},
    "absent": {"present"},
    "available": {"unavailable"},
    "unavailable": {"available"},
    "exist": {"not exist", "disappear"},
    "missing": {"present"},

    # Truth / Certainty
    "true": {"false"},
    "false": {"true"},
    "correct": {"incorrect", "wrong"},
    "incorrect": {"correct"},
    "certain": {"uncertain"},
    "uncertain": {"certain"},
    "likely": {"unlikely"},
    "unlikely": {"likely"},

    # Inclusion / Scope
    "include": {"exclude"},
    "exclude": {"include"},
    "all": {"none"},
    "none": {"all"},
    "always": {"never"},
    "never": {"always"},

    # Enable / Disable
    "enable": {"disable"},
    "disable": {"enable"},
    "allow": {"prevent", "block", "deny"},
    "prevent": {"allow", "enable"},
    "block": {"allow", "permit"},

    # Success / Outcome
    "success": {"failure"},
    "failure": {"success"},
    "win": {"lose"},
    "lose": {"win"},
    "achieve": {"fail"},
    "fail": {"succeed"},

    # Financial / Business
    "profit": {"loss"},
    "loss": {"profit"},
    "revenue": {"loss"},
    "gain": {"loss"},
    "increase": {"decrease"},
    "growth": {"decline"},
    "expansion": {"contraction"},
    "contraction": {"expansion"},

    # Strength / Intensity
    "strong": {"weak"},
    "weak": {"strong"},
    "significant": {"insignificant"},
    "insignificant": {"significant"},

    # Visibility / State
    "visible": {"invisible", "hidden"},
    "hidden": {"visible"},
    "active": {"inactive"},
    "inactive": {"active"},
    "open": {"closed"},
    "closed": {"open"},

    # Agreement / Logic
    "consistent": {"inconsistent"},
    "inconsistent": {"consistent"},
    "agree": {"disagree"},
    "disagree": {"agree"},
    "support": {"oppose"},
    "oppose": {"support"},

    # Risk / Safety
    "safe": {"unsafe", "dangerous"},
    "unsafe": {"safe"},
    "secure": {"insecure"},
    "insecure": {"secure"},

    # Frequency
    "often": {"rarely"},
    "rarely": {"often"},
    "frequent": {"infrequent"},
    "infrequent": {"frequent"}
}

class ContradictionScorer:

    def entity_soft(self, E_c, E_e):
        overlap = len(E_c & E_e) / max(1, len(E_c))
        return ENTITY_SOFT_GAMMA + (1 - ENTITY_SOFT_GAMMA) * overlap
    
    def antonym(self, T_c, T_e):
    # Ensure token-level comparison
        tokens_c = set(T_c)
        tokens_e = set(T_e)

        count = 0

        for w1, antonym_set in ANTONYMS.items():
            if w1 in tokens_c:
                for w2 in antonym_set:
                    if w2 in tokens_e:
                        count += 1

            if w1 in tokens_e:
                for w2 in antonym_set:
                    if w2 in tokens_c:
                        count += 1

        # Normalize (slightly better denominator)
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