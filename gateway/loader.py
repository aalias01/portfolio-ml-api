"""Load isolated FastAPI sub-apps and lazy model weights per portfolio service."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from fastapi import FastAPI

SERVICES_ROOT = Path(__file__).resolve().parent.parent / "services"

_SERVICE_APPS: dict[str, FastAPI] = {}
_MODEL_LOADERS: dict[str, Callable[[], None]] = {}
_MODELS_LOADED: set[str] = set()


def _purge_shared_modules() -> None:
    for key in list(sys.modules):
        if key == "api" or key.startswith("api.") or key == "src" or key.startswith("src."):
            del sys.modules[key]


def _import_service_app(service: str) -> FastAPI:
    root = (SERVICES_ROOT / service).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing service directory: {root}")

    _purge_shared_modules()
    root_str = str(root)
    sys.path.insert(0, root_str)
    try:
        import api.main as main_mod
        import api.predictor as predictor_mod

        if service == "retail":
            _MODEL_LOADERS[service] = predictor_mod.load_all_models
        elif service in {"industrial", "cmapss"}:
            _MODEL_LOADERS[service] = predictor_mod.load_model
        elif service == "hvac":
            _MODEL_LOADERS[service] = predictor_mod.load_scorer
        elif service == "maintenance":
            _MODEL_LOADERS[service] = predictor_mod.load_all
        else:
            raise ValueError(f"Unknown service: {service}")

        return main_mod.app
    finally:
        if sys.path and sys.path[0] == root_str:
            sys.path.pop(0)


def get_service_app(service: str) -> FastAPI:
    if service not in _SERVICE_APPS:
        _SERVICE_APPS[service] = _import_service_app(service)
    return _SERVICE_APPS[service]


def ensure_models_loaded(service: str) -> None:
    if service in _MODELS_LOADED:
        return
    if service not in _MODEL_LOADERS:
        get_service_app(service)
    loader = _MODEL_LOADERS[service]
    loader()
    _MODELS_LOADED.add(service)


def models_loaded(service: str) -> bool:
    return service in _MODELS_LOADED


def mount_all_services(gateway: FastAPI) -> None:
    for service, prefix in SERVICE_MOUNTS.items():
        sub = get_service_app(service)
        gateway.mount(prefix, sub)


SERVICE_MOUNTS: dict[str, str] = {
    "retail": "/retail",
    "industrial": "/industrial",
    "cmapss": "/cmapss",
    "hvac": "/hvac",
    "maintenance": "/maintenance",
}

SERVICE_PREFIX_TO_NAME = {prefix: name for name, prefix in SERVICE_MOUNTS.items()}
