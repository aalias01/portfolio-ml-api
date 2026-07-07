"""Gateway FastAPI app — mounts five portfolio ML APIs under path prefixes."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from gateway.loader import (
    SERVICE_MOUNTS,
    SERVICE_PREFIX_TO_NAME,
    ensure_models_loaded,
    models_loaded,
    mount_all_services,
)

SHARED_ORIGINS = [
    "https://returns.alvinalias.com",
    "https://machine-failure.alvinalias.com",
    "https://turbofan.alvinalias.com",
    "https://hvac.alvinalias.com",
    "https://workorders.alvinalias.com",
    "https://alvinalias.com",
    "https://www.alvinalias.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8124",
    "http://127.0.0.1:8124",
]

app = FastAPI(
    title="Portfolio ML API",
    description=(
        "Shared Hugging Face backend for Alvin Alias portfolio ML demos. "
        "Prefixes: /retail, /industrial, /cmapss, /hvac, /maintenance. "
        "RAG assistant remains on Render."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=SHARED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def lazy_load_models(request: Request, call_next):
    path = request.url.path
    for prefix, service in SERVICE_PREFIX_TO_NAME.items():
        if path == prefix or path.startswith(f"{prefix}/"):
            ensure_models_loaded(service)
            break
    return await call_next(request)


@app.get("/")
def root() -> dict:
    services = {
        name: {
            "prefix": prefix,
            "health": f"{prefix}/health",
            "docs": f"{prefix}/docs",
            "models_loaded": models_loaded(name),
        }
        for name, prefix in SERVICE_MOUNTS.items()
    }
    return {
        "project": "Portfolio ML API",
        "author": "Alvin Alias",
        "services": services,
        "note": "RAG assistant API is not mounted here; see rag.alvinalias.com (Render).",
    }


@app.get("/health")
def health() -> dict:
    return {
        "gateway": "ok",
        "services": {
            name: {
                "prefix": prefix,
                "models_loaded": models_loaded(name),
                "health_url": f"{prefix}/health",
            }
            for name, prefix in SERVICE_MOUNTS.items()
        },
    }


mount_all_services(app)
