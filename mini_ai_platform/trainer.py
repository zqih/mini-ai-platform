from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, List

from .config import storage_root

LogFn = Callable[[str, str], None]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def generate_dataset(samples: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for _ in range(samples):
        x1 = rng.uniform(-2.0, 2.0)
        x2 = rng.uniform(-2.0, 2.0)
        margin = 1.4 * x1 - 1.9 * x2 + 0.25 + rng.gauss(0, 0.25)
        label = 1 if margin > 0 else 0
        rows.append({"x": [x1, x2], "y": label})
    return rows


def evaluate(dataset: List[Dict[str, Any]], weights: List[float], bias: float) -> Dict[str, float]:
    total_loss = 0.0
    correct = 0
    for row in dataset:
        x1, x2 = row["x"]
        y = row["y"]
        pred = sigmoid(weights[0] * x1 + weights[1] * x2 + bias)
        clipped = min(max(pred, 1e-7), 1 - 1e-7)
        total_loss += -(y * math.log(clipped) + (1 - y) * math.log(1 - clipped))
        correct += int((pred >= 0.5) == bool(y))
    return {
        "loss": round(total_loss / len(dataset), 6),
        "accuracy": round(correct / len(dataset), 6),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def artifact_relative(path: Path) -> str:
    return str(path.relative_to(storage_root()))


def run_training(job: Dict[str, Any], log: LogFn) -> Dict[str, Any]:
    """Run a small deterministic classifier training job.

    The trainer is intentionally lightweight so the platform can run on any
    laptop. It still behaves like a real training runtime: it has epochs,
    metrics, checkpoints, a final model artifact, and structured logs.
    """

    job_id = job["id"]
    params = json.loads(job["hyperparameters"])
    epochs = int(params.get("epochs", 8))
    learning_rate = float(params.get("learning_rate", 0.3))
    samples = int(params.get("samples", 256))
    seed = int(params.get("seed", 42))

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if samples < 20:
        raise ValueError("samples must be >= 20")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    job_dir = storage_root() / "artifacts" / job_id
    checkpoint_dir = job_dir / "checkpoints"
    job_dir.mkdir(parents=True, exist_ok=True)

    log("INFO", f"Preparing dataset={job['dataset']} samples={samples} seed={seed}")
    dataset = generate_dataset(samples=samples, seed=seed)
    rng = random.Random(seed + 1)
    weights = [rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)]
    bias = 0.0

    log("INFO", f"Starting training model={job['model_type']} epochs={epochs} lr={learning_rate}")
    for epoch in range(1, epochs + 1):
        grad_w = [0.0, 0.0]
        grad_b = 0.0
        for row in dataset:
            x1, x2 = row["x"]
            y = row["y"]
            pred = sigmoid(weights[0] * x1 + weights[1] * x2 + bias)
            error = pred - y
            grad_w[0] += error * x1
            grad_w[1] += error * x2
            grad_b += error

        scale = 1 / len(dataset)
        weights[0] -= learning_rate * grad_w[0] * scale
        weights[1] -= learning_rate * grad_w[1] * scale
        bias -= learning_rate * grad_b * scale

        metrics = evaluate(dataset, weights, bias)
        log("INFO", f"epoch={epoch}/{epochs} loss={metrics['loss']} accuracy={metrics['accuracy']}")

        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        write_json(
            checkpoint_path,
            {
                "job_id": job_id,
                "epoch": epoch,
                "weights": weights,
                "bias": bias,
                "metrics": metrics,
            },
        )

    final_metrics = evaluate(dataset, weights, bias)
    model_path = job_dir / "model.json"
    metrics_path = job_dir / "metrics.json"
    write_json(
        model_path,
        {
            "job_id": job_id,
            "model_type": job["model_type"],
            "dataset": job["dataset"],
            "weights": weights,
            "bias": bias,
            "threshold": 0.5,
            "input_schema": {"features": ["x1", "x2"]},
        },
    )
    write_json(metrics_path, final_metrics)
    log("INFO", f"Training finished accuracy={final_metrics['accuracy']} loss={final_metrics['loss']}")

    return {
        "metrics": final_metrics,
        "artifacts": [
            {"type": "final_model", "path": model_path, "filename": "model.json"},
            {"type": "metrics", "path": metrics_path, "filename": "metrics.json"},
            {
                "type": "checkpoint",
                "path": checkpoint_dir / f"checkpoint_epoch_{epochs}.json",
                "filename": f"checkpoint_epoch_{epochs}.json",
            },
        ],
    }

