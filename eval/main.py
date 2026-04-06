from judges.prometheus import PrometheusJudge
from judges.mixtral import MixtralJudge
from judges.deepseek import DeepSeekJudge
from judges.ensemble import JudgeEnsemble
from judges.client import BedrockClient
from evaluator.llm.metrics import UnifiedLLMJudge

from evaluator.executor import UnifiedExecutor
from evaluator.deterministic.metrics import DeterministicMetrics
from evaluator.aggregator import Aggregator

from input.retriever import RAGRetriever

from escalator.router import EscalationRouter
from uncertainty.analyzer import UncertaintyAnalyzer
from structuring.claim_reasoner import ClaimReasoner
from evidence.triage import EvidenceTriage
from debate.adjudicator import Adjudicator
from debate.debaters import DebateEngine
from evaluator.deterministic.similarity import Similarity
from evaluator.deterministic.source_types import get_type_weight, is_verifiable

from pipeline import EvaluationPipeline

from sentence_transformers import SentenceTransformer

from datetime import datetime


EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_fn(text):
    return EMBED_MODEL.encode(text).tolist()


def get_client(model_name: str):
    return BedrockClient(model_name)


def main():
    # judges 
    prometheus = PrometheusJudge(
        client=get_client("prometheus-eval/prometheus-7b-v2.0")
    )

    mixtral = MixtralJudge(
        client=get_client("mistralai/Mistral-7B-Instruct-v0.2")
    )

    deepseek = DeepSeekJudge(
        client=get_client("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    )

    ensemble = JudgeEnsemble([prometheus, mixtral])
    llm_judge = UnifiedLLMJudge(ensemble)

    # core components 
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

    router = EscalationRouter()
    uncertainty_analyzer = UncertaintyAnalyzer()

    reasoner = ClaimReasoner(deepseek)
    triage = EvidenceTriage(Similarity())

    debate_engine = DebateEngine(prometheus, mixtral)
    adjudicator = Adjudicator(prometheus)

    # pipeline 
    pipeline = EvaluationPipeline(
        retriever=RAGRetriever(),
        embed_fn=embed_fn,
        reasoner=reasoner,
        triage=triage,
        metric_executor=metric_executor,
        uncertainty_analyzer=uncertainty_analyzer,
        escalation_router=router,
        debate_engine=debate_engine,
        adjudicator=adjudicator,
    )

    # example run 
    claim = "The model significantly improves performance."

    result = pipeline.run(claim)

    print("\n=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()