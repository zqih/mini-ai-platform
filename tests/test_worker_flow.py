from __future__ import annotations

import json


def test_worker_processes_queued_job(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_AI_PLATFORM_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("MINI_AI_PLATFORM_DB", str(tmp_path / "platform.db"))

    from mini_ai_platform.db import create_job, get_job, init_db, list_artifacts, list_logs
    from mini_ai_platform.worker import Worker

    init_db()
    job = create_job(
        name="pytest-job",
        dataset="synthetic-digits",
        model_type="logistic-regression",
        hyperparameters={"epochs": 3, "learning_rate": 0.4, "samples": 100, "seed": 99},
    )

    assert Worker(worker_id="pytest-worker").run_once() is True

    completed = get_job(job["id"])
    assert completed is not None
    assert completed["status"] == "SUCCEEDED"

    metrics = json.loads(completed["metrics"])
    assert metrics["accuracy"] >= 0.75

    logs = list_logs(job["id"])
    assert any("Job succeeded" in row["message"] for row in logs)

    artifacts = list_artifacts(job["id"])
    assert {artifact["type"] for artifact in artifacts} == {"final_model", "metrics", "checkpoint"}

