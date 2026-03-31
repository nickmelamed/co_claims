from metrics.deterministic import DeterministicMetrics


def test_eas():
    m = DeterministicMetrics()
    assert 0 <= m.eas(0) <= 1
    assert m.eas(5) > m.eas(1)


def test_coverage():
    m = DeterministicMetrics()

    claim = {"a", "b", "c"}
    evidence = [{"a", "b"}]

    cov = m.coverage(claim, evidence)

    assert 0 <= cov <= 1
    assert cov == 2 / 3


def test_cms_numeric_bias():
    m = DeterministicMetrics()

    entities = {"accuracy 90%", "latency", "model"}
    score = m.cms(entities)

    assert score > 0  # numeric presence boosts score