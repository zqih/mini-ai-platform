from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from .config import storage_root
from .db import (
    create_job,
    decode_json_field,
    get_artifact,
    get_job,
    init_db,
    list_artifacts,
    list_jobs,
    list_logs,
    scheduler_snapshot,
)
from .schemas import (
    ArtifactResponse,
    ComputeNodeResponse,
    JobCreate,
    JobListResponse,
    JobResponse,
    LogResponse,
    SchedulerResponse,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Mini AI Platform", version="0.2.0", lifespan=lifespan)


def serialize_job(row: Dict[str, Any]) -> JobResponse:
    payload = dict(row)
    payload["hyperparameters"] = decode_json_field(row, "hyperparameters")
    payload["metrics"] = decode_json_field(row, "metrics")
    return JobResponse(**payload)


def serialize_node(row: Dict[str, Any]) -> ComputeNodeResponse:
    payload = dict(row)
    payload["labels"] = decode_json_field(row, "labels")
    return ComputeNodeResponse(**payload)


@app.get("/", response_class=HTMLResponse)
def console() -> str:
    return WEB_CONSOLE_HTML


@app.post("/api/jobs", response_model=JobResponse)
def submit_job(request: JobCreate) -> JobResponse:
    hyperparameters = {
        "epochs": request.epochs,
        "learning_rate": request.learning_rate,
        "samples": request.samples,
        "seed": request.seed,
    }
    job = create_job(
        name=request.name,
        dataset=request.dataset,
        model_type=request.model_type,
        hyperparameters=hyperparameters,
        requested_gpus=request.requested_gpus,
        priority=request.priority,
    )
    return serialize_job(job)


@app.get("/api/jobs", response_model=JobListResponse)
def get_jobs() -> JobListResponse:
    return JobListResponse(jobs=[serialize_job(row) for row in list_jobs()])


@app.get("/api/scheduler", response_model=SchedulerResponse)
def get_scheduler() -> SchedulerResponse:
    snapshot = scheduler_snapshot()
    return SchedulerResponse(
        nodes=[serialize_node(row) for row in snapshot["nodes"]],
        queued_jobs=[serialize_job(row) for row in snapshot["queued_jobs"]],
        running_jobs=[serialize_job(row) for row in snapshot["running_jobs"]],
        total_gpus=snapshot["total_gpus"],
        used_gpus=snapshot["used_gpus"],
        available_gpus=snapshot["available_gpus"],
    )


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job_detail(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return serialize_job(job)


@app.get("/api/jobs/{job_id}/logs", response_model=list[LogResponse])
def get_job_logs(job_id: str, after_id: int = 0) -> list[LogResponse]:
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return [LogResponse(**row) for row in list_logs(job_id, after_id=after_id)]


@app.get("/api/jobs/{job_id}/artifacts", response_model=list[ArtifactResponse])
def get_job_artifacts(job_id: str) -> list[ArtifactResponse]:
    if get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="job not found")
    return [ArtifactResponse(**row) for row in list_artifacts(job_id)]


@app.get("/api/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str) -> FileResponse:
    artifact = get_artifact(artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = (storage_root() / artifact["path"]).resolve()
    if storage_root() not in path.parents and path != storage_root():
        raise HTTPException(status_code=400, detail="invalid artifact path")
    if not path.exists():
        raise HTTPException(status_code=404, detail="artifact file missing")
    return FileResponse(path, filename=artifact["filename"])


WEB_CONSOLE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mini AI Platform</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dee4;
      --text: #24292f;
      --muted: #57606a;
      --blue: #0969da;
      --green: #1a7f37;
      --red: #cf222e;
      --yellow: #9a6700;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    h1 { font-size: 20px; margin: 0; }
    main {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 16px;
      padding: 16px 24px 24px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 { font-size: 15px; margin: 0 0 12px; }
    label { display: block; margin: 12px 0 6px; color: var(--muted); font-size: 12px; }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    button {
      border: 1px solid var(--blue);
      background: var(--blue);
      color: white;
      border-radius: 6px;
      padding: 8px 12px;
      font-weight: 600;
      cursor: pointer;
    }
    button.secondary {
      background: white;
      color: var(--blue);
    }
    table {
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 600; }
    tr[data-selected="true"] { background: #ddf4ff; }
    .toolbar { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: #f6f8fa;
    }
    .QUEUED { color: var(--yellow); }
    .RUNNING { color: var(--blue); }
    .SUCCEEDED { color: var(--green); }
    .FAILED { color: var(--red); }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
    }
    pre {
      margin: 0;
      min-height: 260px;
      max-height: 460px;
      overflow: auto;
      background: #0d1117;
      color: #c9d1d9;
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
    }
    .artifact a { color: var(--blue); text-decoration: none; }
    .node {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      margin-top: 8px;
    }
    .meter {
      height: 8px;
      border-radius: 999px;
      background: #eaeef2;
      overflow: hidden;
      margin-top: 6px;
    }
    .meter > span {
      display: block;
      height: 100%;
      background: var(--blue);
    }
    .muted { color: var(--muted); }
    @media (max-width: 900px) {
      main, .split { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Mini AI Platform</h1>
    <span class="badge">Local training control plane</span>
  </header>
  <main>
    <section>
      <h2>Create Training Job</h2>
      <form id="job-form">
        <label for="name">Name</label>
        <input id="name" value="synthetic-training-job" />
        <label for="epochs">Epochs</label>
        <input id="epochs" type="number" min="1" max="200" value="8" />
        <label for="learning_rate">Learning rate</label>
        <input id="learning_rate" type="number" min="0.01" step="0.05" value="0.3" />
        <label for="samples">Samples</label>
        <input id="samples" type="number" min="20" max="10000" value="256" />
        <label for="seed">Seed</label>
        <input id="seed" type="number" value="42" />
        <label for="requested_gpus">Requested GPUs</label>
        <input id="requested_gpus" type="number" min="1" max="8" value="1" />
        <label for="priority">Priority</label>
        <input id="priority" type="number" min="0" max="100" value="0" />
        <div class="toolbar" style="margin-top:16px">
          <button type="submit">Submit Job</button>
          <button class="secondary" type="button" id="refresh">Refresh</button>
        </div>
      </form>
      <p class="muted">Start a worker in another terminal with <code>./scripts/run_worker.sh</code>.</p>
      <h2 style="margin-top:18px">Scheduler</h2>
      <div id="scheduler" class="muted">Loading resources...</div>
    </section>
    <div class="split">
      <section>
        <div class="toolbar">
          <h2 style="margin-right:auto">Jobs</h2>
          <span class="muted" id="job-count">0 jobs</span>
        </div>
        <table>
          <thead>
            <tr>
              <th style="width:160px">ID</th>
              <th>Name</th>
              <th style="width:80px">GPUs</th>
              <th style="width:110px">Node</th>
              <th style="width:120px">Status</th>
              <th style="width:130px">Accuracy</th>
            </tr>
          </thead>
          <tbody id="jobs"></tbody>
        </table>
      </section>
      <section>
        <h2>Selected Job</h2>
        <div id="detail" class="muted">No job selected.</div>
        <h2 style="margin-top:18px">Artifacts</h2>
        <div id="artifacts" class="muted">No artifacts yet.</div>
      </section>
      <section style="grid-column:1 / -1">
        <h2>Logs</h2>
        <pre id="logs">No logs yet.</pre>
      </section>
    </div>
  </main>
<script>
let selectedJobId = null;

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function statusBadge(status) {
  return `<span class="badge ${status}">${status}</span>`;
}

async function refreshScheduler() {
  const scheduler = await api('/api/scheduler');
  const target = document.getElementById('scheduler');
  target.innerHTML = `
    <div><strong>${scheduler.used_gpus}/${scheduler.total_gpus}</strong> GPUs used</div>
    <div class="muted">${scheduler.queued_jobs.length} queued, ${scheduler.running_jobs.length} running</div>
    ${scheduler.nodes.map(node => {
      const pct = node.total_gpus ? Math.round((node.used_gpus / node.total_gpus) * 100) : 0;
      return `<div class="node">
        <div><strong>${node.name}</strong> <span class="badge">${node.status}</span></div>
        <div>${node.used_gpus}/${node.total_gpus} GPUs used</div>
        <div class="meter"><span style="width:${pct}%"></span></div>
      </div>`;
    }).join('')}
  `;
}

async function refreshJobs() {
  const data = await api('/api/jobs');
  const tbody = document.getElementById('jobs');
  document.getElementById('job-count').textContent = `${data.jobs.length} jobs`;
  tbody.innerHTML = data.jobs.map(job => {
    const accuracy = job.metrics && job.metrics.accuracy !== undefined ? job.metrics.accuracy : '-';
    return `<tr data-job-id="${job.id}" data-selected="${job.id === selectedJobId}">
      <td><button class="secondary" data-open="${job.id}">${job.id}</button></td>
      <td>${job.name}</td>
      <td>${job.requested_gpus}</td>
      <td>${job.allocated_node_id || '-'}</td>
      <td>${statusBadge(job.status)}</td>
      <td>${accuracy}</td>
    </tr>`;
  }).join('');
  tbody.querySelectorAll('button[data-open]').forEach(button => {
    button.addEventListener('click', () => selectJob(button.dataset.open));
  });
  if (!selectedJobId && data.jobs.length) await selectJob(data.jobs[0].id);
  if (selectedJobId) await refreshSelected();
  await refreshScheduler();
}

async function selectJob(jobId) {
  selectedJobId = jobId;
  await refreshSelected();
  await refreshJobs();
}

async function refreshSelected() {
  if (!selectedJobId) return;
  const job = await api(`/api/jobs/${selectedJobId}`);
  const detail = document.getElementById('detail');
  detail.innerHTML = `
    <div><strong>${job.name}</strong></div>
    <div>ID: ${job.id}</div>
    <div>Status: ${statusBadge(job.status)}</div>
    <div>GPUs: ${job.requested_gpus}</div>
    <div>Priority: ${job.priority}</div>
    <div>Node: ${job.allocated_node_id || '-'}</div>
    <div>Dataset: ${job.dataset}</div>
    <div>Model: ${job.model_type}</div>
    <div>Created: ${job.created_at}</div>
  `;

  const logs = await api(`/api/jobs/${selectedJobId}/logs`);
  document.getElementById('logs').textContent = logs.length
    ? logs.map(row => `[${row.timestamp}] ${row.level} ${row.message}`).join('\\n')
    : 'No logs yet.';

  const artifacts = await api(`/api/jobs/${selectedJobId}/artifacts`);
  document.getElementById('artifacts').innerHTML = artifacts.length
    ? artifacts.map(a => `<div class="artifact">${a.type}: <a href="/api/artifacts/${a.id}/download">${a.filename}</a> <span class="muted">(${a.size_bytes} bytes)</span></div>`).join('')
    : '<span class="muted">No artifacts yet.</span>';
}

document.getElementById('job-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const payload = {
    name: document.getElementById('name').value,
    epochs: Number(document.getElementById('epochs').value),
    learning_rate: Number(document.getElementById('learning_rate').value),
    samples: Number(document.getElementById('samples').value),
    seed: Number(document.getElementById('seed').value),
    requested_gpus: Number(document.getElementById('requested_gpus').value),
    priority: Number(document.getElementById('priority').value)
  };
  const job = await api('/api/jobs', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  selectedJobId = job.id;
  await refreshJobs();
});

document.getElementById('refresh').addEventListener('click', refreshJobs);
setInterval(refreshJobs, 3000);
refreshJobs().catch(err => {
  document.getElementById('logs').textContent = err.message;
});
</script>
</body>
</html>
"""
