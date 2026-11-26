import pytest


def test_ping_does_not_open_database(monkeypatch):
    import run_pipeline

    for name in (
        "S3_BUCKET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "BIIGLE_API_EMAIL",
        "BIIGLE_API_TOKEN",
        "ZOONIVERSE_USER",
        "ZOONIVERSE_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    def fail_if_created(*args, **kwargs):
        raise AssertionError("Ping must not create the pipeline database")

    monkeypatch.setattr(run_pipeline, "DatabaseManager", fail_if_created)
    monkeypatch.setattr(run_pipeline.sys, "argv", ["run_pipeline.py", "--ping"])

    with pytest.raises(SystemExit) as result:
        run_pipeline.main()

    assert result.value.code == 1
