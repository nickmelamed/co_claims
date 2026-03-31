import math
from itertools import combinations

EPS = 1e-6

HEDGE_TERMS = {
    "may", "might", "could", "can", "suggests", "appears",
    "likely", "potentially", "approximately", "generally", "often"
}


class ClaimMetrics:

    def __init__(self, contradiction_scorer):
        self.contradiction = contradiction_scorer

    def hls(self, claim_f):
        tokens = claim_f["tokens"]
        if not tokens:
            return 1.0

        hedge_count = sum(1 for t in tokens if t in HEDGE_TERMS)
        return 1 - (hedge_count / len(tokens))

    def cms(self, entities):
        weights = []
        for e in entities:
            if any(char.isdigit() for char in e):
                weights.append(1.0)
            else:
                weights.append(0.0)
        return sum(weights) / max(1, len(entities))

    def cscope(self, entities):
        n = len(entities)
        return 1 / (1 + math.log(1 + n))

    def lcs(self, claim_f):
        doc = claim_f["doc"]
        props = [sent.text for sent in doc.sents]

        if len(props) < 2:
            return 1.0

        scores = []
        for p1, p2 in combinations(props, 2):
            f1 = {"tokens": set(p1.lower().split()), "entities": claim_f["entities"]}
            f2 = {"tokens": set(p2.lower().split()), "entities": claim_f["entities"]}
            scores.append(self.contradiction.score(f1, f2))

        return 1 - (sum(scores) / (len(scores) + EPS))