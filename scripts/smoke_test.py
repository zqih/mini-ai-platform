from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="mini-ai-platform-smoke-"))
    try:
        os.environ["MINI_AI_PLATFORM_STORAGE"] = str(tmpdir / "storage")
        os.environ["MINI_AI_PLATFORM_DB"] = str(tmpdir / "platform.db")

        from mini_ai_platform.db import create_job, get_job, init_db, list_artifacts, list_logs
        from mini_ai_platform.worker import Worker

        init_db()
        job = create_job(
            name="smoke-test",
            dataset="synthetic-digits",
            model_type="logistic-regression",
            hyperparameters={"epochs": 4, "learning_rate": 0.35, "samples": 120, "seed": 7},
        )
        assert job["status"] == "QUEUED"

        worker = Worker(worker_id="smoke-worker")
        assert worker.run_once() is True

        completed = get_job(job["id"])
        assert completed is not None
        assert completed["status"] == "SUCCEEDED", completed
        metrics = json.loads(completed["metrics"])
        assert metrics["accuracy"] >= 0.8, metrics

        logs = list_logs(job["id"])
        assert any("epoch=4/4" in row["message"] for row in logs)

        artifacts = list_artifacts(job["id"])
        artifact_names = {row["filename"] for row in artifacts}
        assert {"model.json", "metrics.json", "checkpoint_epoch_4.json"} <= artifact_names

        print("smoke test passed")
        print(f"job_id={job['id']} accuracy={metrics['accuracy']} artifacts={len(artifacts)}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
