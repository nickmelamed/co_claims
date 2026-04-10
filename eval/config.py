from eval.pipeline import EvaluationPipeline
from eval.judges.prometheus import PrometheusJudge
from eval.judges.mixtral import MixtralJudge
from eval.judges.deepseek import DeepSeekJudge
from eval.judges.ensemble import JudgeEnsemble
from eval.judges.client import BedrockClient
from eval.evaluator.llm.metrics import UnifiedLLMJudge

from eval.structuring.entity_resolver import EntityResolver
from eval.evaluator.deterministic.extractor import FeatureExtractor

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

_EMBED_MODEL = None

def get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBED_MODEL


def embed_fn(text):
    return get_embed_model().encode(text).tolist()

def embed_batch(texts):
    return get_embed_model().encode(texts, 
                                    batch_size=32,
                                    show_progress_bar=False).tolist()

def get_client(model_id):
    return BedrockClient(model_id)

def build_pipeline():

    # Judges
    prometheus = PrometheusJudge(client=get_client("anthropic.claude-3-haiku-20240307-v1:0")) #TODO: find replacement model
    mixtral = MixtralJudge(client=get_client("mistral.mistral-7b-instruct-v0:2"))
    deepseek = DeepSeekJudge(client=get_client("deepseek.r1-v1:0"))

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

    extractor = FeatureExtractor()

    entity_resolver = EntityResolver(
        extractor=extractor,
        reasoner=reasoner # can set to none to disable LLM 
    )


    triage = EvidenceTriage(Similarity())
    router = EscalationRouter()
    uncertainty = UncertaintyAnalyzer()

    debate_engine = DebateEngine(prometheus, mixtral)
    adjudicator = Adjudicator(prometheus)

    return EvaluationPipeline(
        embed_fn=embed_fn,
        embed_batch_fn=embed_batch,
        entity_resolver=entity_resolver,
        reasoner=reasoner,
        triage=triage,
        metric_executor=metric_executor,
        uncertainty_analyzer=uncertainty,
        escalation_router=router,
        debate_engine=debate_engine,
        adjudicator=adjudicator,
    )