from judges.prometheus import PrometheusJudge
from judges.mixtral import MixtralJudge
from judges.ensemble import JudgeEnsemble

from metrics.deterministic import DeterministicMetrics

from metrics.executor import UnifiedExecutor
from escalator.router import EscalationRouter
from pipeline import EvaluationPipeline

# TODO: Replace with actual client 
class DummyClient:
    def chat(self):
        pass


def get_client():
    import os
    import httpx

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY before running the evaluator.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

    class _ChatCompletionsAPI:
        def __init__(self, client):
            self._client = client

        def create(self, **payload):
            response = self._client.post(
                "/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            message = type("Message", (), {"content": data["choices"][0]["message"]["content"]})()
            choice = type("Choice", (), {"message": message})()
            return type("ChatCompletionResponse", (), {"choices": [choice]})()

    class _ChatAPI:
        def __init__(self, client):
            self.completions = _ChatCompletionsAPI(client)

    class _Client:
        def __init__(self):
            self._http = httpx.Client(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
            self.chat = _ChatAPI(self)

        def post(self, path, **kwargs):
            return self._http.post(path, **kwargs)

    return _Client()


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

    # Metrics
    ensemble = JudgeEnsemble(prometheus, mixtral)
    deterministic = DeterministicMetrics()

    # Components
    metric_executor = UnifiedExecutor(ensemble)
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
