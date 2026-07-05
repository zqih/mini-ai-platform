from __future__ import annotations


def test_scheduler_waits_when_no_node_has_enough_gpus(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_AI_PLATFORM_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("MINI_AI_PLATFORM_DB", str(tmp_path / "platform.db"))

    from mini_ai_platform.db import (
        claim_next_job,
        create_job,
        get_node,
        init_db,
        release_job_resources,
        reset_cluster,
    )

    init_db()
    reset_cluster([{"id": "tiny-node", "name": "Tiny Node", "total_gpus": 4}])

    large_job = create_job(
        name="large",
        dataset="synthetic-digits",
        model_type="logistic-regression",
        hyperparameters={"epochs": 1, "learning_rate": 0.3, "samples": 40, "seed": 1},
        requested_gpus=4,
    )
    waiting_job = create_job(
        name="waiting",
        dataset="synthetic-digits",
        model_type="logistic-regression",
        hyperparameters={"epochs": 1, "learning_rate": 0.3, "samples": 40, "seed": 2},
        requested_gpus=2,
    )

    claimed = claim_next_job("scheduler-test-worker")
    assert claimed is not None
    assert claimed["id"] == large_job["id"]
    assert claimed["allocated_node_id"] == "tiny-node"

    node = get_node("tiny-node")
    assert node is not None
    assert node["used_gpus"] == 4
    assert node["available_gpus"] == 0

    assert claim_next_job("scheduler-test-worker") is None

    release_job_resources(large_job["id"])
    next_claimed = claim_next_job("scheduler-test-worker")
    assert next_claimed is not None
    assert next_claimed["id"] == waiting_job["id"]
    assert next_claimed["allocated_node_id"] == "tiny-node"


def test_scheduler_uses_best_fit_node(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_AI_PLATFORM_STORAGE", str(tmp_path / "storage"))
    monkeypatch.setenv("MINI_AI_PLATFORM_DB", str(tmp_path / "platform.db"))

    from mini_ai_platform.db import claim_next_job, create_job, init_db, reset_cluster

    init_db()
    reset_cluster(
        [
            {"id": "node-a", "name": "Node A", "total_gpus": 8},
            {"id": "node-b", "name": "Node B", "total_gpus": 4},
            {"id": "node-c", "name": "Node C", "total_gpus": 2},
        ]
    )

    job = create_job(
        name="best-fit",
        dataset="synthetic-digits",
        model_type="logistic-regression",
        hyperparameters={"epochs": 1, "learning_rate": 0.3, "samples": 40, "seed": 3},
        requested_gpus=2,
    )

    claimed = claim_next_job("best-fit-worker")
    assert claimed is not None
    assert claimed["id"] == job["id"]
    assert claimed["allocated_node_id"] == "node-c"
