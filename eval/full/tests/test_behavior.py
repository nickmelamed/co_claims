from judges.ensemble import JudgeEnsemble

def test_support_vs_contradiction_direction(mock_prometheus, mock_mixtral):
    ensemble = JudgeEnsemble(mock_prometheus, mock_mixtral)

    metrics, _, _ = ensemble.evaluate("claim", "supportive evidence")

    assert metrics["ESS"] >= metrics["ECS"]