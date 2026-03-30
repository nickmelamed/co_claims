def test_pipeline_runs():
    from main import main

    # Just ensure no crash
    try:
        main()
        assert True
    except Exception as e:
        assert False, str(e)