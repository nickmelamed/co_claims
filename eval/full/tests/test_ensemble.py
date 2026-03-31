from judges.ensemble import JudgeEnsemble


def test_ensemble_mean_variance(mock_prometheus, mock_mixtral):
    ensemble = JudgeEnsemble(mock_prometheus, mock_mixtral)

    metrics, variance, raw = ensemble.evaluate("claim", "evidence")

    assert "ESS" in metrics
    assert "ESS" in variance

    assert 0 <= metrics["ESS"] <= 1
    assert variance["ESS"] >= 0