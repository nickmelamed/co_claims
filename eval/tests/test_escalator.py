from escalator.router import EscalationRouter


def test_escalation_variance():
    router = EscalationRouter()

    metrics = {"ESS": 0.6, "ECS": 0.4}
    uncertainty = {
        "ESS_var": 0.1,
        "ECS_var": 0.1,
        "CMS_var": 0.0,
        "LCS_var": 0.0,
        "HLS_var": 0.0,
        "mean_variance": 0.1
    }

    decision = router.decide(metrics, uncertainty, 3)

    assert decision["decision"] == "escalate"