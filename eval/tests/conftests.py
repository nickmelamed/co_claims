import pytest


class MockLLM:
    def __init__(self, outputs):
        self.outputs = outputs

    def evaluate(self, claim, evidence):
        return self.outputs


@pytest.fixture
def mock_prometheus():
    return MockLLM({
        "ESS": 0.8,
        "ECS": 0.2,
        "CMS": 0.7,
        "LCS": 0.9,
        "HLS": 0.85,
        "confidence": 0.9
    })


@pytest.fixture
def mock_mixtral():
    return MockLLM({
        "ESS": 0.6,
        "ECS": 0.3,
        "CMS": 0.6,
        "LCS": 0.8,
        "HLS": 0.75,
        "confidence": 0.8
    })