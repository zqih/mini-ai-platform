from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_api_submit_job_then_worker_completes(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_AI_PLATFORM_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("MINI_AI_PLATFORM_DB", str(tmp_path / "platform.db"))

    import mini_ai_platform.main as main_module
    from mini_ai_platform.worker import Worker

    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/jobs",
            json={"name": "api-test", "epochs": 2, "learning_rate": 0.4, "samples": 80},
        )
        assert response.status_code == 200
        job = response.json()
        assert job["status"] == "QUEUED"
        assert job["requested_gpus"] == 1

        assert Worker(worker_id="api-test-worker").run_once() is True

        job_response = client.get(f"/api/jobs/{job['id']}")
        assert job_response.status_code == 200
        completed = job_response.json()
        assert completed["status"] == "SUCCEEDED"
        assert completed["metrics"]["accuracy"] >= 0.75
        assert completed["allocated_node_id"] == "node-c"
        assert completed["resources_released_at"] is not None

        scheduler_response = client.get("/api/scheduler")
        assert scheduler_response.status_code == 200
        scheduler = scheduler_response.json()
        assert scheduler["total_gpus"] == 14
        assert scheduler["used_gpus"] == 0
        assert len(scheduler["nodes"]) == 3

        logs_response = client.get(f"/api/jobs/{job['id']}/logs")
        assert logs_response.status_code == 200
        assert any("Job succeeded" in row["message"] for row in logs_response.json())

        artifacts_response = client.get(f"/api/jobs/{job['id']}/artifacts")
        assert artifacts_response.status_code == 200
        assert {artifact["type"] for artifact in artifacts_response.json()} == {
            "checkpoint",
            "final_model",
            "metrics",
        }
