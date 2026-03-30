# TODO determine if we want LLM on top of thresholding for evidence

class EvidenceTriage:
    def __init__(self, sim, judge):
        self.sim = sim
        self.judge = judge

    def filter(self, claim_embedding, evidence_list):
        filtered = []

        for e in evidence_list:
            r = self.sim.relevance(claim_embedding, e["embedding"])

            if r < 0.3:
                continue

            # additional LLM labeling for evidence
            label = self.judge.evaluate(f"""
            Classify evidence relevance:
            Claim: {e['claim']}
            Evidence: {e['text']}
            
            Output JSON:
            {{ "relevant": true/false }}
            """)

            if label.get("relevant", True):
                filtered.append(e)

        return filtered