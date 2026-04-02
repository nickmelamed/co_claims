from judges.prometheus import PrometheusJudge
from judges.mixtral import MixtralJudge
from judges.deepseek import DeepSeekJudge
from judges.ensemble import JudgeEnsemble
from judges.client import LocalLLMClient

from evaluator.executor import MetricExecutor
from escalator.router import EscalationRouter
from uncertainty.analyzer import UncertaintyAnalyzer
from structuring.claim_reasoner import ClaimReasoner
from evidence.triage import EvidenceTriage
from debate.adjudicator import Adjudicator
from debate.debaters import DebateEngine
from evaluator.aggregator import Aggregator
from input.similarity import Similarity

from pipeline import EvaluationPipeline

from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# TODO: replace with real retriever 
class DummyRetriever:
    def retrieve(self, claim, extra=False):
        return [
            {
                "text": "The model improves accuracy by 5% on benchmark datasets."
            },
            {
                "text": "Some studies show no improvement in performance."
            }
        ]

def embed_fn(text):
    return EMBED_MODEL.encode(text).tolist()


def main():

    def get_client(model_name: str):
        return LocalLLMClient(model_name)

    # Judges
    prometheus = PrometheusJudge(client=get_client("prometheus-eval/prometheus-7b-v2.0"))
    mixtral = MixtralJudge(client=get_client("mistralai/Mistral-7B-Instruct-v0.2"))
    deepseek = DeepSeekJudge(client=get_client("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))

    ensemble = JudgeEnsemble([prometheus, mixtral])

    # Components
    metric_executor = MetricExecutor(ensemble)
    router = EscalationRouter()
    uncertainty_analyzer = UncertaintyAnalyzer()

    reasoner = ClaimReasoner(deepseek)
    triage = EvidenceTriage(Similarity())
    aggregator = Aggregator()
    debate_engine = DebateEngine(prometheus, mixtral)
    adjudicator = Adjudicator(prometheus)

    pipeline = EvaluationPipeline(
        retriever=DummyRetriever(),
        embed_fn=embed_fn,
        reasoner=reasoner,
        triage=triage,
        metric_executor=metric_executor,
        uncertainty_analyzer=uncertainty_analyzer,
        escalation_router=router,
        debate_engine=debate_engine,
        adjudicator=adjudicator,
        aggregator=aggregator
    )

    claim = "The model significantly improves performance."

    result = pipeline.run(claim)

    print("\n=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()