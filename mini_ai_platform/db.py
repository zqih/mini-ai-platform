from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import db_path, ensure_runtime_dirs


TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELED"}
DEFAULT_COMPUTE_NODES = [
    {"id": "node-a", "name": "Node A", "total_gpus": 8, "labels": {"zone": "local-a"}},
    {"id": "node-b", "name": "Node B", "total_gpus": 4, "labels": {"zone": "local-b"}},
    {"id": "node-c", "name": "Node C", "total_gpus": 2, "labels": {"zone": "local-c"}},
]


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
                metrics TEXT,
                requested_gpus INTEGER NOT NULL DEFAULT 1,
                priority INTEGER NOT NULL DEFAULT 0,
                allocated_node_id TEXT,
                scheduled_at TEXT,
                resources_released_at TEXT
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

            CREATE TABLE IF NOT EXISTS compute_nodes (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                total_gpus INTEGER NOT NULL,
                used_gpus INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'READY',
                labels TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        _ensure_column(conn, "jobs", "requested_gpus", "INTEGER NOT NULL DEFAULT 1")
        _ensure_column(conn, "jobs", "priority", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "jobs", "allocated_node_id", "TEXT")
        _ensure_column(conn, "jobs", "scheduled_at", "TEXT")
        _ensure_column(conn, "jobs", "resources_released_at", "TEXT")
        _seed_default_nodes(conn)


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def _seed_default_nodes(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) AS count FROM compute_nodes").fetchone()["count"]
    if count:
        return
    now = utc_now()
    for node in DEFAULT_COMPUTE_NODES:
        conn.execute(
            """
            INSERT INTO compute_nodes (
                id, name, total_gpus, used_gpus, status, labels, created_at, updated_at
            )
            VALUES (?, ?, ?, 0, 'READY', ?, ?, ?)
            """,
            (
                node["id"],
                node["name"],
                node["total_gpus"],
                json.dumps(node["labels"], sort_keys=True),
                now,
                now,
            ),
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
    requested_gpus: int = 1,
    priority: int = 0,
) -> Dict[str, Any]:
    init_db()
    if requested_gpus < 1:
        raise ValueError("requested_gpus must be >= 1")
    job_id = uuid.uuid4().hex[:12]
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id, name, dataset, model_type, hyperparameters, status,
                created_at, updated_at, requested_gpus, priority
            )
            VALUES (?, ?, ?, ?, ?, 'QUEUED', ?, ?, ?, ?)
            """,
            (
                job_id,
                name,
                dataset,
                model_type,
                json.dumps(hyperparameters, sort_keys=True),
                now,
                now,
                requested_gpus,
                priority,
            ),
        )
    append_log(job_id, "INFO", f"Job queued requested_gpus={requested_gpus} priority={priority}")
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
    resources_released_at: Optional[str] = None,
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
    if resources_released_at is not None:
        fields.append("resources_released_at = ?")
        values.append(resources_released_at)

    values.append(job_id)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)


def list_nodes() -> List[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *,
                   total_gpus - used_gpus AS available_gpus
            FROM compute_nodes
            ORDER BY id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_node(node_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *,
                   total_gpus - used_gpus AS available_gpus
            FROM compute_nodes
            WHERE id = ?
            """,
            (node_id,),
        ).fetchone()
    return row_to_dict(row)


def reset_cluster(nodes: Optional[List[Dict[str, Any]]] = None) -> None:
    init_db()
    node_specs = nodes or DEFAULT_COMPUTE_NODES
    now = utc_now()
    with connect() as conn:
        conn.execute("DELETE FROM compute_nodes")
        for node in node_specs:
            conn.execute(
                """
                INSERT INTO compute_nodes (
                    id, name, total_gpus, used_gpus, status, labels, created_at, updated_at
                )
                VALUES (?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    node["id"],
                    node.get("name", node["id"]),
                    int(node["total_gpus"]),
                    node.get("status", "READY"),
                    json.dumps(node.get("labels", {}), sort_keys=True),
                    now,
                    now,
                ),
            )


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


def _best_fit_node(nodes: List[sqlite3.Row], requested_gpus: int) -> Optional[sqlite3.Row]:
    candidates = [
        node
        for node in nodes
        if node["status"] == "READY" and node["total_gpus"] - node["used_gpus"] >= requested_gpus
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda node: (node["total_gpus"] - node["used_gpus"] - requested_gpus, node["id"]),
    )[0]


def claim_next_job(worker_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        jobs = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = 'QUEUED'
            ORDER BY priority DESC, created_at ASC
            """
        ).fetchall()
        if not jobs:
            conn.execute("COMMIT")
            return None

        nodes = conn.execute("SELECT * FROM compute_nodes ORDER BY id ASC").fetchall()
        row: Optional[sqlite3.Row] = None
        node: Optional[sqlite3.Row] = None
        for candidate in jobs:
            requested_gpus = int(candidate["requested_gpus"])
            candidate_node = _best_fit_node(nodes, requested_gpus)
            if candidate_node is not None:
                row = candidate
                node = candidate_node
                break

        if row is None or node is None:
            conn.execute("COMMIT")
            return None

        now = utc_now()
        requested_gpus = int(row["requested_gpus"])
        conn.execute(
            """
            UPDATE jobs
            SET status = 'RUNNING',
                started_at = ?,
                updated_at = ?,
                worker_id = ?,
                allocated_node_id = ?,
                scheduled_at = ?,
                resources_released_at = NULL
            WHERE id = ? AND status = 'QUEUED'
            """,
            (now, now, worker_id, node["id"], now, row["id"]),
        )
        conn.execute(
            """
            UPDATE compute_nodes
            SET used_gpus = used_gpus + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (requested_gpus, now, node["id"]),
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


def release_job_resources(job_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        if row["allocated_node_id"] is None or row["resources_released_at"] is not None:
            conn.execute("COMMIT")
            return dict(row)

        now = utc_now()
        requested_gpus = int(row["requested_gpus"])
        conn.execute(
            """
            UPDATE compute_nodes
            SET used_gpus = MAX(used_gpus - ?, 0),
                updated_at = ?
            WHERE id = ?
            """,
            (requested_gpus, now, row["allocated_node_id"]),
        )
        conn.execute(
            """
            UPDATE jobs
            SET resources_released_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, job_id),
        )
        conn.execute("COMMIT")
        return get_job(job_id)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def scheduler_snapshot() -> Dict[str, Any]:
    nodes = list_nodes()
    jobs = list_jobs(limit=500)
    queued = [job for job in jobs if job["status"] == "QUEUED"]
    running = [job for job in jobs if job["status"] == "RUNNING"]
    total_gpus = sum(int(node["total_gpus"]) for node in nodes)
    used_gpus = sum(int(node["used_gpus"]) for node in nodes)
    return {
        "nodes": nodes,
        "queued_jobs": queued,
        "running_jobs": running,
        "total_gpus": total_gpus,
        "used_gpus": used_gpus,
        "available_gpus": total_gpus - used_gpus,
    }


def decode_json_field(row: Dict[str, Any], field: str) -> Dict[str, Any]:
    raw = row.get(field)
    if not raw:
        return {}
    return json.loads(raw)
