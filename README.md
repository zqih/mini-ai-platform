# Mini AI Platform

A small, runnable AI training platform for learning AI systems concepts.

The first version implements the core platform flow:

```text
Web Console / API
-> Job metadata in SQLite
-> DB-backed queue
-> Worker process
-> Training runtime
-> Logs, metrics, checkpoints, model artifacts
```

It intentionally starts without Docker, Kubernetes, Redis, or GPUs. Those are later phases. This version focuses on making the training job lifecycle visible and debuggable.

## Features

- Submit training jobs from a browser or REST API
- Persist jobs, logs, metrics, and artifacts in SQLite
- Run an independent worker process that claims queued jobs
- Simulate a GPU cluster with Node A/B/C resources
- Schedule jobs by requested GPU count with best-fit placement
- Keep jobs queued when no node has enough free GPUs
- Release simulated GPU resources when jobs finish or fail
- Train a lightweight synthetic classifier with checkpoint output
- View job status and logs in a simple web console
- Download model, metrics, and checkpoint artifacts

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1
./scripts/run_api.sh

# Terminal 2
./scripts/run_worker.sh
```

Open http://127.0.0.1:8000.

## Smoke Test

```bash
source .venv/bin/activate
python scripts/smoke_test.py
```

## API Example

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"name":"first-run","epochs":8,"learning_rate":0.4,"samples":320,"requested_gpus":2}'
```

Then query:

```bash
curl http://127.0.0.1:8000/api/jobs
curl http://127.0.0.1:8000/api/scheduler
curl http://127.0.0.1:8000/api/jobs/<job_id>/logs
curl http://127.0.0.1:8000/api/jobs/<job_id>/artifacts
```

## Project Layout

```text
mini_ai_platform/
  main.py      # FastAPI app and Web Console
  db.py        # SQLite metadata store and DB-backed queue
  worker.py    # Worker loop that claims and runs jobs
  trainer.py   # Training runtime and artifact generation
  schemas.py   # API schemas

storage/
  artifacts/   # Runtime training outputs
  logs/        # Reserved for file log storage
  models/      # Reserved for promoted model registry artifacts
  datasets/    # Reserved for uploaded datasets
```

## Roadmap

1. Replace the synthetic trainer with a PyTorch MNIST/CIFAR trainer.
2. Run each job inside a Docker container.
3. Add richer scheduling policies: priority queues, preemption, quotas, and fragmentation reports.
4. Add model registry and deploy-to-inference flow.
5. Move job execution to Kubernetes Jobs.
