import pytest
from evaluator import ClaimEvaluator
from sentence_transformers import SentenceTransformer
import numpy as np
from evaluator.metrics import EvidenceMetrics, SourceMetrics, ClaimMetrics
from evaluator.contradiction import ContradictionScorer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def embed_fn(text):
    return MODEL.encode(text)

@pytest.fixture
def evaluator():
    return ClaimEvaluator(embed_fn)


def test_strong_support(evaluator):
    claim = "The model improves accuracy by 20%"
    evidence = [{
        "text": "The model improves accuracy by 20%",
        "embedding": embed_fn("The model improves accuracy by 20%"),
        "source": "https://arxiv.org"
    }]

    result = evaluator.evaluate(claim, evidence)

    # good amount of support clearly 
    assert result["ESS"] > 0.6

    # might be some contradiction with embedding, but not much 
    assert result["ECS"] < 0.2


def test_contradiction(evaluator):
    claim = "The model improves accuracy"
    evidence = [{
        "text": "The model decreases accuracy",
        "embedding": embed_fn("The model decreases accuracy"),
        "source": "https://arxiv.org"
    }]

    result = evaluator.evaluate(claim, evidence)

    # definitely a contradiction, but wordwise may not be the highest contradiction
    assert result["ECS"] > 0.4


def test_vague_claim(evaluator):
    claim = "This model is very powerful"
    evidence = [{
        "text": "The model is widely used",
        "embedding": embed_fn("The model is widely used"),
        "source": "https://blog.com"
    }]

    result = evaluator.evaluate(claim, evidence)

    # no measurability to this claim 
    assert result["CMS"] == 0

def test_new_company(evaluator):
    claim = "Our startup model improves accuracy by 15%"

    evidence = [{
        "text": "Initial internal results show a 15% improvement in accuracy.",
        "embedding": embed_fn("Initial internal results show a 15% improvement in accuracy."),
        "source": "https://startup.ai"
    }]

    result = evaluator.evaluate(claim, evidence)

    # Should have SOME support
    assert result["ESS"] > 0.3

    # Credibility should not be maxed out
    assert result["Credibility"] < 0.8

def test_negation_contradiction(evaluator):
    claim = "The model improves accuracy"

    evidence = [{
        "text": "The model does not improve accuracy in most cases.",
        "embedding": embed_fn("The model does not improve accuracy in most cases."),
        "source": "https://arxiv.org"
    }]

    result = evaluator.evaluate(claim, evidence)

    # Should detect contradiction
    assert result["ECS"] > 0.2

    # Support should be low
    assert result["ESS"] < 0.5


def test_evidence_metrics():
    em = EvidenceMetrics()

    supports = [0.8, 0.6, 0.9]
    relevances = [0.9, 0.7, 0.8]

    assert 0 <= em.ess(supports, relevances) <= 1
    assert 0 <= em.ecs(supports, relevances) <= 1
    assert 0 <= em.eas(3) <= 1
    assert 0 <= em.eags(supports) <= 1


def test_source_metrics():
    sm = SourceMetrics()

    domains = ["a.com", "b.com", "a.com"]
    flags = [1, 1, 0]

    assert sm.srs(domains) == 2 / 3
    assert sm.evs(flags) == sum(flags) / len(flags)


def test_claim_metrics():
    contradiction = ContradictionScorer()
    cm = ClaimMetrics(contradiction)

    claim_f = {
        "tokens": {"model", "may", "improve", "accuracy"},
        "entities": {"model", "accuracy"},
        "doc": type("doc", (), {
            "sents": [type("s", (), {"text": "model improves accuracy"})()]
        })()
    }

    assert 0 <= cm.hls(claim_f) <= 1
    assert 0 <= cm.cms(claim_f["entities"]) <= 1
    assert 0 <= cm.cscope(claim_f["entities"]) <= 1
    assert 0 <= cm.lcs(claim_f) <= 1


def test_eags_behavior():
    em = EvidenceMetrics()

    consistent = [0.9, 0.9, 0.9]
    inconsistent = [0.1, 0.9, 0.5]

    assert em.eags(consistent) > em.eags(inconsistent)