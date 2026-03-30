import pytest

from metrics.executor import MetricExecutor
from judges.ensemble import JudgeEnsemble


# Mock Judge 
class MockJudge:
    def __init__(self, value):
        self.value = value

    def evaluate(self, prompt):
        return {"score": self.value, "confidence": 1.0}


@pytest.fixture
def executor():
    # deterministic mock ensemble
    judge1 = MockJudge(0.8)
    judge2 = MockJudge(0.6)

    ensemble = JudgeEnsemble(judge1, judge2)

    return MetricExecutor(ensemble)


def test_metric_output_structure(executor):
    claim = "Model improves accuracy"
    evidence = [{"text": "Accuracy improves by 5%"}]
    relevances = [1.0]

    result = executor.evaluate(claim, evidence, relevances)

    assert "metrics" in result
    assert "uncertainty" in result

    for key in ["ESS", "ECS", "CMS", "LCS", "HLS"]:
        assert key in result["metrics"]


def test_metric_ranges(executor):
    claim = "Model improves accuracy"
    evidence = [{"text": "Accuracy improves by 5%"}]
    relevances = [1.0]

    result = executor.evaluate(claim, evidence, relevances)

    for v in result["metrics"].values():
        assert 0.0 <= v <= 1.0


def test_uncertainty_present(executor):
    claim = "Model improves accuracy"
    evidence = [{"text": "Accuracy improves by 5%"}]
    relevances = [1.0]

    result = executor.evaluate(claim, evidence, relevances)

    assert "mean_variance" in result["uncertainty"]