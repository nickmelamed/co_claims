from eval.pipeline import EvaluationPipeline
from eval.judges.prometheus import PrometheusJudge
from eval.judges.mixtral import MixtralJudge
from eval.judges.deepseek import DeepSeekJudge
from eval.judges.ensemble import JudgeEnsemble
from eval.judges.client import BedrockClient
from eval.evaluator.llm.metrics import UnifiedLLMJudge

from eval.evaluator.executor import UnifiedExecutor
from eval.evaluator.deterministic.metrics import DeterministicMetrics
from eval.evaluator.aggregator import Aggregator

from eval.escalator.router import EscalationRouter
from eval.uncertainty.analyzer import UncertaintyAnalyzer
from eval.structuring.claim_reasoner import ClaimReasoner
from eval.evidence.triage import EvidenceTriage
from eval.debate.adjudicator import Adjudicator
from eval.debate.debaters import DebateEngine
from eval.evaluator.deterministic.similarity import Similarity
from eval.evaluator.deterministic.source_types import get_type_weight, is_verifiable

from sentence_transformers import SentenceTransformer


EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def embed_fn(text):
    return EMBED_MODEL.encode(text).tolist()

def get_client(model_id):
    return BedrockClient(model_id)

def build_pipeline():

    # Judges
    prometheus = PrometheusJudge(client=get_client("prometheus-eval/prometheus-7b-v2.0"))
    mixtral = MixtralJudge(client=get_client("mistralai/Mistral-7B-Instruct-v0.2"))
    deepseek = DeepSeekJudge(client=get_client("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))

    ensemble = JudgeEnsemble([prometheus, mixtral])
    llm_judge = UnifiedLLMJudge(ensemble)

    # Deterministic
    deterministic = DeterministicMetrics(
        type_weight_fn=get_type_weight,
        verifiable_fn=is_verifiable
    )
    aggregator = Aggregator()

    metric_executor = UnifiedExecutor(
        llm_judge=llm_judge,
        deterministic_metrics=deterministic,
        aggregator=aggregator
    )

    # Pipeline components
    reasoner = ClaimReasoner(deepseek)
    triage = EvidenceTriage(Similarity())
    router = EscalationRouter()
    uncertainty = UncertaintyAnalyzer()

    debate_engine = DebateEngine(prometheus, mixtral)
    adjudicator = Adjudicator(prometheus)

    return EvaluationPipeline(
        embed_fn=embed_fn,
        reasoner=reasoner,
        triage=triage,
        metric_executor=metric_executor,
        uncertainty_analyzer=uncertainty,
        escalation_router=router,
        debate_engine=debate_engine,
        adjudicator=adjudicator,
    )