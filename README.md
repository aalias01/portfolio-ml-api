---
title: Portfolio ML API
sdk: docker
app_port: 7860
fullWidth: true
header: mini
suggested_hardware: cpu-basic
---

# Portfolio ML API

Single Hugging Face Space hosting five portfolio ML backends behind path prefixes.

| Prefix | Demo | Frontend |
|--------|------|----------|
| `/retail` | Retail Returns Intelligence | https://returns.alvinalias.com |
| `/industrial` | Industrial Failure Classification | https://machine-failure.alvinalias.com |
| `/cmapss` | C-MAPSS RUL | https://turbofan.alvinalias.com |
| `/hvac` | HVAC Equipment Health | https://hvac.alvinalias.com |
| `/maintenance` | Maintenance Work-Order NLP | https://workorders.alvinalias.com |

Gateway: `/` · `/health` · `/docs`

RAG Engineering Assistant stays on Render (`rag.alvinalias.com`), not mounted here.

Source code is synced from `aalias01/portfolio-ml-api` on GitHub. Canonical project repos live under the portfolio workspace.
