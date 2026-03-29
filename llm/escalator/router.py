# TODO determine proper variance threshold for escalation 

UNCERTAINTY_THRESHOLD = 0.02

class EscalationRouter:
    def decide(self, uncertainty):
        if uncertainty["mean_variance"] < UNCERTAINTY_THRESHOLD:
            return "accept"
        return "escalate"