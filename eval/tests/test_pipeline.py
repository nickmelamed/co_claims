def test_pipeline_runs():
    from main import build_pipeline

    pipeline = build_pipeline()

    result = pipeline.run("Test claim")

    assert "metrics" in result
    assert "credibility" in result