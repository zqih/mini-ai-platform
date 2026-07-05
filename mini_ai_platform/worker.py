from __future__ import annotations

import argparse
import socket
import time
import traceback
from pathlib import Path
from typing import Optional

from .config import storage_root
from .db import (
    append_log,
    claim_next_job,
    create_artifact,
    init_db,
    update_job,
    utc_now,
)
from .trainer import artifact_relative, run_training


class Worker:
    def __init__(self, worker_id: Optional[str] = None) -> None:
        self.worker_id = worker_id or f"{socket.gethostname()}-{int(time.time())}"

    def run_once(self) -> bool:
        job = claim_next_job(self.worker_id)
        if job is None:
            return False

        job_id = job["id"]
        append_log(job_id, "INFO", f"Worker {self.worker_id} claimed job")
        try:
            result = run_training(job, lambda level, message: append_log(job_id, level, message))
            for artifact in result["artifacts"]:
                path = Path(artifact["path"])
                create_artifact(
                    job_id=job_id,
                    artifact_type=artifact["type"],
                    relative_path=artifact_relative(path),
                    filename=artifact["filename"],
                    size_bytes=path.stat().st_size,
                )
                append_log(job_id, "INFO", f"Artifact saved: {artifact['filename']}")

            update_job(
                job_id,
                status="SUCCEEDED",
                finished_at=utc_now(),
                metrics=result["metrics"],
            )
            append_log(job_id, "INFO", "Job succeeded")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            append_log(job_id, "ERROR", error)
            append_log(job_id, "ERROR", traceback.format_exc(limit=5))
            update_job(
                job_id,
                status="FAILED",
                finished_at=utc_now(),
                error_message=error,
            )
        return True

    def run_forever(self, poll_interval: float = 2.0) -> None:
        init_db()
        print(f"Worker {self.worker_id} watching queue at {storage_root()}")
        while True:
            processed = self.run_once()
            if not processed:
                time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini AI Platform worker")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args()

    worker = Worker(worker_id=args.worker_id)
    if args.once:
        processed = worker.run_once()
        print("processed=1" if processed else "processed=0")
    else:
        worker.run_forever(poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()

