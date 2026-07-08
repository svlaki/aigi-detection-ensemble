"""Pydantic request/response schemas for the demo API."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MemberResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    raw_score: float
    calibrated_score: float


class EnsembleResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    confidence: float
    verdict: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: str
    confidence: float
    mode: str
    members: list[MemberResult]
    ensemble_methods: list[EnsembleResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    d3qe_available: bool
    device: str
