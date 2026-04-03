from metrics.executor import UnifiedExecutor
from metrics.deterministic import DeterministicMetrics
from metrics.deterministic import Metrics
from judges.ensemble import UnifiedEnsemble


def test_executor_combines_metrics(mock_prometheus, mock_mixtral):
    ensemble = UnifiedEnsemble(mock_prometheus, mock_mixtral)
    det = DeterministicMetrics(Metrics())

    executor = UnifiedExecutor(ensemble, det)

    claim = "Model improves accuracy"
    claim_f = {"entities": {"accuracy"}}

    evidence = [
        {"text": "Accuracy improves", "entities": {"accuracy"}}
    ]

    domains = ["arxiv.org"]

    result = executor.evaluate(claim, claim_f, evidence, domains)

    assert "metrics" in result
    assert "uncertainty" in result

    # check merged metrics
    for key in ["ESS", "ECS", "CMS", "EAS", "SRS"]:
        assert key in result["metrics"]