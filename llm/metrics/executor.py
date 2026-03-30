from .ess import ESS_LLM
from .ecs import ECS_LLM
from .cms import CMS_LLM
from .lcs import LCS_LLM
from .hls import HLS_LLM


class MetricExecutor:
    def __init__(self, ensemble):
        self.ess = ESS_LLM(ensemble)
        self.ecs = ECS_LLM(ensemble)
        self.cms = CMS_LLM(ensemble)
        self.lcs = LCS_LLM(ensemble)
        self.hls = HLS_LLM(ensemble)

    def evaluate(self, claim, evidence_list, relevances):
        """
        Runs all metric evaluations
        """

        ESS = self.ess.score(claim, evidence_list, relevances)
        ECS = self.ecs.score(claim, evidence_list, relevances)
        CMS = self.cms.score(claim)
        LCS = self.lcs.score(claim)
        HLS = self.hls.score(claim)

        return {
            "ESS": ESS,
            "ECS": ECS,
            "CMS": CMS,
            "LCS": LCS,
            "HLS": HLS
        }