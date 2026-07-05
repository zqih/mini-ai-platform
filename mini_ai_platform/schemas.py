from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    name: str = Field(default="synthetic-training-job", min_length=1, max_length=80)
    dataset: str = Field(default="synthetic-digits")
    model_type: str = Field(default="logistic-regression")
    epochs: int = Field(default=8, ge=1, le=200)
    learning_rate: float = Field(default=0.3, gt=0, le=5)
    samples: int = Field(default=256, ge=20, le=10000)
    seed: int = Field(default=42)
    requested_gpus: int = Field(default=1, ge=1, le=8)
    priority: int = Field(default=0, ge=0, le=100)


class JobResponse(BaseModel):
    id: str
    name: str
    dataset: str
    model_type: str
    hyperparameters: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    requested_gpus: int = 1
    priority: int = 0
    allocated_node_id: Optional[str] = None
    scheduled_at: Optional[str] = None
    resources_released_at: Optional[str] = None


class LogResponse(BaseModel):
    id: int
    job_id: str
    timestamp: str
    level: str
    message: str


class ArtifactResponse(BaseModel):
    id: str
    job_id: str
    type: str
    path: str
    filename: str
    size_bytes: int
    created_at: str


class JobListResponse(BaseModel):
    jobs: List[JobResponse]


class ComputeNodeResponse(BaseModel):
    id: str
    name: str
    total_gpus: int
    used_gpus: int
    available_gpus: int
    status: str
    labels: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class SchedulerResponse(BaseModel):
    nodes: List[ComputeNodeResponse]
    queued_jobs: List[JobResponse]
    running_jobs: List[JobResponse]
    total_gpus: int
    used_gpus: int
    available_gpus: int
