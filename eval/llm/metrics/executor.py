from .ess import ESS_LLM
from .ecs import ECS_LLM
from .cms import CMS_LLM
from .lcs import LCS_LLM
from .hls import HLS_LLM

# TODO: determine if need mean variance calculation (not sure we do...)

class MetricExecutor:
    def __init__(self, ensemble):
        self.ess = ESS_LLM(ensemble)
        self.ecs = ECS_LLM(ensemble)
        self.cms = CMS_LLM(ensemble)
        self.lcs = LCS_LLM(ensemble)
        self.hls = HLS_LLM(ensemble)

    def evaluate(self, claim, evidence_list, relevances):
        ESS, ESS_var = self.ess.score(claim, evidence_list, relevances)
        ECS, ECS_var = self.ecs.score(claim, evidence_list, relevances)

        CMS, CMS_var = self.cms.score(claim)
        LCS, LCS_var = self.lcs.score(claim)
        HLS, HLS_var = self.hls.score(claim)

        return {
            "metrics": {
                "ESS": ESS,
                "ECS": ECS,
                "CMS": CMS,
                "LCS": LCS,
                "HLS": HLS
            },
            "variances": {
                "ESS_var": ESS_var,
                "ECS_var": ECS_var,
                "CMS_var": CMS_var,
                "LCS_var": LCS_var,
                "HLS_var": HLS_var
            }
        }