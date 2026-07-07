"""
api/main.py — FastAPI application for HVAC Equipment Health Scoring.

Endpoints:
    GET  /              → project description and links
    GET  /health        → API health check + scorer status
    GET  /units         → summary scores for all units (dashboard view)
    POST /score         → score a single HVAC unit sensor snapshot
    POST /score/batch   → score multiple units in one request

Start locally:
    uvicorn api.main:app --reload

Deploy:
    Render reads render.yaml (buildCommand + startCommand configured there).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    DemoScenariosResponse,
    HealthResponse,
    SensorReading,
    ScoreResponse,
    UnitListResponse,
)
import api.predictor as predictor
from src.scorer import TIER_CUTOFFS


# ---------------------------------------------------------------------------
# Lifespan — load scorer at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_scorer()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HVAC Equipment Health Scoring API",
    description=(
        "Scores HVAC unit health (0–100) from sensor data using Isolation Forest "
        "anomaly detection and domain-engineered features (COP, ΔT, load ratio). "
        "Built by an engineer who spent 3 years designing HVAC systems at Rheem Manufacturing."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — update origins after Vercel deploy
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://hvac.alvinalias.com",               # canonical demo (Primary)
        "https://hvac-equipment-health.vercel.app",  # legacy, 308-redirects to subdomain
        # Portfolio landing live-model playground (alvinalias.com).
        "https://alvinalias.com", "https://www.alvinalias.com",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=JSONResponse)
def root():
    return {
        "project": "HVAC Equipment Health Scoring",
        "description": (
            "End-to-end anomaly detection for HVAC systems using domain-engineered features "
            "(COP, ΔT, load ratio) built from refrigeration physics knowledge."
        ),
        "endpoints": {
            "health":     "GET /health",
            "units":      "GET /units",
            "demo":       "GET /demo-readings",
            "score":      "POST /score",
            "batch":      "POST /score/batch",
            "docs":       "GET /docs",
        },
        "github": "https://github.com/aalias01/hvac-equipment-health",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    scorer = predictor.get_scorer()
    return HealthResponse(
        status="ok" if predictor.is_ready() else "degraded",
        scorer_loaded=predictor.is_ready(),
        feature_count=len(scorer.feature_names) if scorer else 0,
        contamination=scorer.contamination if scorer else 0.05,
        tiers=TIER_CUTOFFS,
        fleet_snapshot=True,
    )


@app.get("/units", response_model=UnitListResponse)
def get_units():
    """
    Return a summary of health scores for all units in the training corpus.
    Sorted worst-first (lowest health score at top) for triage prioritization.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Scorer not loaded. Run notebook 03 to train and save models.",
        )
    return predictor.get_all_units()


@app.get("/demo-readings", response_model=DemoScenariosResponse)
def get_demo_readings():
    """
    Return curated, complete historical readings for the frontend demo.

    These scenarios avoid placeholder rolling features and are scored against
    each source unit's own baseline when posted to /score.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Scorer not loaded. Run notebook 03 to train and save models.",
        )
    try:
        return predictor.get_demo_readings()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/score", response_model=ScoreResponse)
def score_unit(
    reading: SensorReading,
    shap: bool = Query(default=True, description="Include SHAP explanations in response"),
):
    """
    Score a single HVAC unit sensor snapshot.

    Returns a 0–100 health score, health tier (healthy/monitor/warning/critical),
    anomaly flag, and top SHAP factors explaining the score.
    """
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Scorer not loaded. Run notebook 03 to train and save models.",
        )
    started = perf_counter()
    status = 200
    try:
        return predictor.score_single(reading, include_shap=shap)
    except Exception as e:
        status = 500
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed_ms = (perf_counter() - started) * 1000
        print(f"[api] route=/score status={status} n=1 shap={shap} ms={elapsed_ms:.1f}")


@app.post("/score/batch", response_model=List[ScoreResponse])
def score_batch(
    readings: List[SensorReading],
    shap: bool = Query(default=False, description="Include SHAP (slower for large batches)"),
):
    """Score multiple units in a single request. SHAP disabled by default for speed."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Scorer not loaded.")
    if len(readings) > 500:
        raise HTTPException(status_code=400, detail="Batch limited to 500 readings per request.")
    started = perf_counter()
    status = 200
    try:
        results = []
        for r in readings:
            try:
                results.append(predictor.score_single(r, include_shap=shap))
            except Exception:
                results.append(
                    ScoreResponse(
                        building_id=r.building_id,
                        health_score=0.0,
                        health_tier="critical",
                        anomaly_flag=1,
                        iforest_score=0.0,
                    )
                )
        return results
    except Exception:
        status = 500
        raise
    finally:
        elapsed_ms = (perf_counter() - started) * 1000
        print(f"[api] route=/score/batch status={status} n={len(readings)} shap={shap} ms={elapsed_ms:.1f}")
