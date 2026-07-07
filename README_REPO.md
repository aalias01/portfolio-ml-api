# Portfolio ML API

Shared Hugging Face backend for five portfolio ML demos (retail, industrial, cmapss, hvac, maintenance).

## Quick links

- Space URL: https://alvinalias-portfolio-ml-api.hf.space
- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Your manual steps:** [YOUR_STEPS.md](YOUR_STEPS.md)
- Manual HF push: [PUSH_TO_HF.md](PUSH_TO_HF.md)

## Sync from source projects

After changing API code or models in any `active/<project>` repo:

```bash
bash scripts/sync_from_portfolio.sh
git add services/
git commit -m "Sync services from portfolio projects"
git push
```

## Local run (optional)

```bash
pip install -r requirements.txt
uvicorn gateway.main:app --reload --port 7860
```

First hit to each prefix loads that service's models (can take tens of seconds).
