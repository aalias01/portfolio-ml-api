from __future__ import annotations
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.schemas import SensorReading, PredictResponse, HealthResponse
import api.predictor as predictor


logger = logging.getLogger("industrial_failure.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_model()
    yield


app = FastAPI(
    title="Industrial Failure Classification API",
    description="Predicts machine failure probability from sensor readings. "
                "Threshold tuned for business cost (FN=$50K, FP=$2K).",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", "http://127.0.0.1:8080",
        "http://localhost:8124", "http://127.0.0.1:8124",
        "https://machine-failure.alvinalias.com",    # canonical demo (Primary)
        "https://industrial-failure-classification.vercel.app",  # legacy, 308-redirects to subdomain
        # Portfolio landing live-model playground (alvinalias.com).
        "https://alvinalias.com", "https://www.alvinalias.com",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_predict_timing(request: Request, call_next):
    if request.url.path != "/predict":
        return await call_next(request)

    started = time.perf_counter()
    status = 500
    shap = request.query_params.get("shap", "true")
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("predict status=%s shap=%s ms=%.1f", status, shap, elapsed_ms)


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "project": "Industrial Failure Classification",
        "endpoints": {"health": "GET /health", "predict": "POST /predict", "docs": "GET /docs"},
        "github": "https://github.com/aalias01/industrial-failure-classification",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    clf = predictor._clf
    return HealthResponse(
        status="ok" if predictor.is_ready() else "degraded",
        model_loaded=predictor.is_ready(),
        model_type=clf.model_type if clf else "none",
        optimal_threshold=clf.optimal_threshold if clf else 0.5,
        risk_zones=predictor.risk_zones(),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    reading: SensorReading,
    shap: bool = Query(default=True, description="Include SHAP explanation"),
):
    """Predict failure probability from sensor readings."""
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Run notebooks first.")
    try:
        return predictor.predict(reading, include_shap=shap)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
