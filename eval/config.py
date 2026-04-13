from eval.pipeline import EvaluationPipeline
from eval.judges.qwen import QwenJudge
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

def build_pipeline(mode="full", retrieve_fn=None):

    # embeddings 
    embed_fn_local = embed_fn
    embed_batch_local = embed_batch

    # deterministic 
    deterministic = DeterministicMetrics(
        type_weight_fn=get_type_weight,
        verifiable_fn=is_verifiable
    )
    aggregator = Aggregator()

    # judges 
    qwen = QwenJudge(client=get_client("qwen.qwen3-32b-v1:0"))

    if mode == "single_llm":
        llm_judge = UnifiedLLMJudge(qwen)

    elif mode == "full":
        mixtral = MixtralJudge(client=get_client("mistral.mistral-7b-instruct-v0:2"))
        ensemble = JudgeEnsemble([qwen, mixtral])
        llm_judge = UnifiedLLMJudge(ensemble)

    else:
        llm_judge = None  # baseline

    # executor 
    executor = UnifiedExecutor(
        llm_judge=llm_judge,
        deterministic_metrics=deterministic,
        aggregator=aggregator,
        use_llm=(mode != "baseline")
    )

    # pipeline 
    deepseek = DeepSeekJudge(client=get_client("deepseek.r1-v1:0"))
    reasoner = ClaimReasoner(deepseek)

    extractor = FeatureExtractor()

    entity_resolver = EntityResolver(
        extractor=extractor,
        reasoner=reasoner
    )

    triage = EvidenceTriage(Similarity())
    router = EscalationRouter()
    uncertainty = UncertaintyAnalyzer()

    debate_engine = DebateEngine(qwen, mixtral)  
    adjudicator = Adjudicator(qwen)

    return EvaluationPipeline(
        embed_fn=embed_fn_local,
        embed_batch_fn=embed_batch_local,
        entity_resolver=entity_resolver,
        reasoner=reasoner,
        triage=triage,
        metric_executor=executor,
        uncertainty_analyzer=uncertainty,
        escalation_router=router,
        debate_engine=debate_engine,
        adjudicator=adjudicator,
        retrieve_fn=retrieve_fn,
        mode=mode  
    )