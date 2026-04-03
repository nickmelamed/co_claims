from metrics.llm import UnifiedLLMJudge


class FakeModel:
    def evaluate(self, prompt):
        return """
        <json>
        {
            "ESS": 0.7,
            "ECS": 0.2,
            "CMS": 0.6,
            "LCS": 0.9,
            "HLS": 0.8,
            "confidence": 0.9
        }
        </json>
        """


def test_unified_llm_output_structure():
    judge = UnifiedLLMJudge(FakeModel())

    result = judge.evaluate("claim", "evidence")

    for k in ["ESS", "ECS", "CMS", "LCS", "HLS"]:
        assert k in result
        assert 0 <= result[k] <= 1