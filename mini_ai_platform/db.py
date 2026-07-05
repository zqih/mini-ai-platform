from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import db_path, ensure_runtime_dirs


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    ensure_runtime_dirs()
    conn = sqlite3.connect(db_path(), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def row_to_dict(row: sqlite3.Row | None) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return dict(row)


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                dataset TEXT NOT NULL,
                model_type TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                worker_id TEXT,
                error_message TEXT,
                metrics TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status_created
                ON jobs(status, created_at);

            CREATE TABLE IF NOT EXISTS job_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_job_logs_job_id_id
                ON job_logs(job_id, id);

            CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                type TEXT NOT NULL,
                path TEXT NOT NULL,
                filename TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_artifacts_job_id
                ON artifacts(job_id);
            """
        )


def reset_db() -> None:
    path = db_path()
    if path.exists():
        path.unlink()
    wal = Path(str(path) + "-wal")
    shm = Path(str(path) + "-shm")
    for extra in [wal, shm]:
        if extra.exists():
            extra.unlink()
    init_db()


def create_job(
    *,
    name: str,
    dataset: str,
    model_type: str,
    hyperparameters: Dict[str, Any],
) -> Dict[str, Any]:
    init_db()
    job_id = uuid.uuid4().hex[:12]
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, name, dataset, model_type, hyperparameters, status,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'QUEUED', ?, ?)
            """,
            (
                job_id,
                name,
                dataset,
                model_type,
                json.dumps(hyperparameters, sort_keys=True),
                now,
                now,
            ),
        )
    append_log(job_id, "INFO", "Job queued")
    job = get_job(job_id)
    assert job is not None
    return job


def list_jobs(limit: int = 100) -> List[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row_to_dict(row)


def update_job(
    job_id: str,
    *,
    status: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    worker_id: Optional[str] = None,
    error_message: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    fields: List[str] = ["updated_at = ?"]
    values: List[Any] = [utc_now()]
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if started_at is not None:
        fields.append("started_at = ?")
        values.append(started_at)
    if finished_at is not None:
        fields.append("finished_at = ?")
        values.append(finished_at)
    if worker_id is not None:
        fields.append("worker_id = ?")
        values.append(worker_id)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    if metrics is not None:
        fields.append("metrics = ?")
        values.append(json.dumps(metrics, sort_keys=True))

    values.append(job_id)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)


def append_log(job_id: str, level: str, message: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO job_logs (job_id, timestamp, level, message)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, utc_now(), level.upper(), message),
        )


def list_logs(job_id: str, after_id: int = 0, limit: int = 500) -> List[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM job_logs
            WHERE job_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (job_id, after_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def create_artifact(
    *,
    job_id: str,
    artifact_type: str,
    relative_path: str,
    filename: str,
    size_bytes: int,
) -> Dict[str, Any]:
    artifact_id = uuid.uuid4().hex[:12]
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO artifacts (
                id, job_id, type, path, filename, size_bytes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (artifact_id, job_id, artifact_type, relative_path, filename, size_bytes, now),
        )
        row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
    return dict(row)


def list_artifacts(job_id: str) -> List[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM artifacts
            WHERE job_id = ?
            ORDER BY created_at ASC
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
    return row_to_dict(row)


def claim_next_job(worker_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'QUEUED'
            ORDER BY created_at ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None

        now = utc_now()
        conn.execute(
            """
            UPDATE jobs
            SET status = 'RUNNING',
                started_at = ?,
                updated_at = ?,
                worker_id = ?
            WHERE id = ? AND status = 'QUEUED'
            """,
            (now, now, worker_id, row["id"]),
        )
        conn.execute("COMMIT")
        claimed = get_job(row["id"])
        assert claimed is not None
        return claimed
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def decode_json_field(row: Dict[str, Any], field: str) -> Dict[str, Any]:
    raw = row.get(field)
    if not raw:
        return {}
    return json.loads(raw)

