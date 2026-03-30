from escalator.router import EscalationRouter


def test_escalation_trigger_variance():
    router = EscalationRouter(var_threshold=0.05)

    metrics = {"ESS": 0.7, "ECS": 0.2}
    uncertainty = {
        "ESS_var": 0.1,
        "ECS_var": 0.1,
        "CMS_var": 0.0,
        "LCS_var": 0.0,
        "HLS_var": 0.0,
        "mean_variance": 0.1
    }

    decision = router.decide(metrics, uncertainty, evidence_count=3)

    assert decision["decision"] == "escalate"


def test_escalation_low_variance():
    router = EscalationRouter(var_threshold=0.05)

    metrics = {"ESS": 0.9, "ECS": 0.1}
    uncertainty = {
        "ESS_var": 0.01,
        "ECS_var": 0.01,
        "CMS_var": 0.01,
        "LCS_var": 0.01,
        "HLS_var": 0.01,
        "mean_variance": 0.01
    }

    decision = router.decide(metrics, uncertainty, evidence_count=3)

    assert decision["decision"] == "accept"