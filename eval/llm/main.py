from judges.prometheus import PrometheusJudge
from judges.mixtral import MixtralJudge
from judges.ensemble import JudgeEnsemble

from metrics.executor import MetricExecutor
from escalator.router import EscalationRouter
from pipeline import EvaluationPipeline

# TODO: Replace with actual client 
class DummyClient:
    def chat(self):
        pass


def get_client():
    # TODO: replace with real client initialization
    return None


# TODO: replace with real retriever 
class DummyRetriever:
    def retrieve(self, claim, extra=False):
        return [
            {
                "text": "The model improves accuracy by 5% on benchmark datasets.",
                "embedding": [0.1, 0.2, 0.3]
            },
            {
                "text": "Some studies show no improvement in performance.",
                "embedding": [0.2, 0.1, 0.4]
            }
        ]


# TODO: replace with real embedding function 
def embed_fn(text):
    return [0.1, 0.2, 0.3]


def main():
    client = get_client()

    # Judges
    prometheus = PrometheusJudge(client)
    mixtral = MixtralJudge(client)

    ensemble = JudgeEnsemble(prometheus, mixtral)

    # Components
    metric_executor = MetricExecutor(ensemble)
    router = EscalationRouter()

    # Minimal placeholders (replace with real ones)
    reasoner = type("Reasoner", (), {"structure": lambda self, x: {"claim": x}})()
    triage = type("Triage", (), {
        "filter": lambda self, emb, ev: ev,
        "sim": type("Sim", (), {
            "relevance": lambda self, a, b: 0.9
        })()
    })()

    aggregator = type("Agg", (), {
        "credibility": lambda self, e, c, n: (e + c) / 2
    })()

    pipeline = EvaluationPipeline(
        retriever=DummyRetriever(),
        embed_fn=embed_fn,
        reasoner=reasoner,
        triage=triage,
        metric_executor=metric_executor,
        uncertainty_analyzer=None,  # handled inside executor now
        escalation_router=router,
        aggregator=aggregator
    )

    claim = "The model significantly improves performance."

    result = pipeline.run(claim)

    print("\n=== FINAL RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()