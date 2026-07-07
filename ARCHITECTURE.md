# Portfolio ML API — architecture

## Problem

Hugging Face free tier limits concurrent **cpu-basic** Spaces. Five independent Docker Spaces compete for quota and block each other from starting.

## Solution

One Space (`alvinalias/portfolio-ml-api`) mounts five existing FastAPI apps under path prefixes. One running container serves all portfolio ML demos except RAG.

## Layout

```
portfolio_ml_api/
  gateway/
    main.py       # CORS, lazy load middleware, root + /health
    loader.py     # Isolated imports, model loaders, mounts
  services/
    retail/       api/ src/ models/
    industrial/   api/ src/ models/
    cmapss/       api/ models/
    hvac/         api/ src/ models/
    maintenance/  api/ src/ models/
  scripts/
    sync_from_portfolio.sh
```

## Routing

| Service | Mount prefix | Example health |
|---------|--------------|----------------|
| Retail Returns | `/retail` | `/retail/health` |
| Industrial Failure | `/industrial` | `/industrial/health` |
| C-MAPSS RUL | `/cmapss` | `/cmapss/health` |
| HVAC Health | `/hvac` | `/hvac/health` |
| Maintenance NLP | `/maintenance` | `/maintenance/health` |

Public URL: `https://alvinalias-portfolio-ml-api.hf.space`

## Import isolation

Each service uses top-level `api` and `src` packages. `loader.py` purges `api.*` / `src.*` from `sys.modules` before each import, then keeps the mounted `FastAPI` app objects. Route handlers retain references to the correct predictor modules.

## Lazy model loading

Mounted sub-app lifespans do not run. Gateway middleware calls each service's predictor load function on the first request to that prefix:

- retail → `load_all_models`
- industrial, cmapss → `load_model`
- hvac → `load_scorer`
- maintenance → `load_all`

## Path patches

Synced copies patch `Path("models")` in predictors to `Path(__file__).resolve().parent.parent / "models"` so artifacts resolve regardless of process cwd.

## CORS

Gateway allows the five custom demo domains, `alvinalias.com`, localhost dev ports, and `*.vercel.app`.

## What stays separate

- **RAG Engineering Assistant** — Render (`rag.alvinalias.com`)
- **Frontends** — Vercel on custom domains; API base URLs point at shared HF prefix
- **Canonical code** — still lives in each project's GitHub repo; `sync_from_portfolio.sh` copies serving subsets here

## Supersedes

Per-repo `deploy/hf_space/` sync workflows for retail, industrial, cmapss, hvac, and maintenance are obsolete once this Space is verified. Pause (do not delete) the old individual HF Spaces to free quota.
