# Your steps (cannot be automated here)

Complete these in order after the local `portfolio_ml_api` folder is ready.

## 1. Create GitHub repo

- Name: `portfolio-ml-api`
- Owner: `aalias01`
- Public, no README (this folder is the initial commit)

```bash
cd "active/portfolio_ml_api"
git init
git add .
git commit -m "Initial shared portfolio ML API gateway"
git branch -M main
git remote add origin https://github.com/aalias01/portfolio-ml-api.git
git push -u origin main
```

## 2. Create Hugging Face Space

- URL: https://huggingface.co/new-space
- Owner: `alvinalias`
- Name: `portfolio-ml-api`
- SDK: **Docker**
- Hardware: **CPU basic**
- Visibility: Public

You do not need to upload files manually if GitHub Actions will sync (step 3).

## 3. Add GitHub secret

In `aalias01/portfolio-ml-api` → Settings → Secrets → Actions:

- `HF_TOKEN` — Hugging Face write token with access to Spaces

Push to `main` (or run **Sync Portfolio ML API HF Space** workflow manually) to deploy.

## 4. Pause old Spaces (free quota)

Before starting the shared Space, pause these if any are Running:

- `alvinalias/retail-returns-intelligence`
- `alvinalias/industrial-failure`
- `alvinalias/cmapss-rul`
- `alvinalias/hvac-health`
- `alvinalias/maintenance-nlp`

Paused is not enough if quota is stuck: **Restart** the shared Space after pausing another.

## 5. Verify shared Space

```bash
curl -s https://alvinalias-portfolio-ml-api.hf.space/ | python3 -m json.tool
curl -s https://alvinalias-portfolio-ml-api.hf.space/retail/health
curl -s https://alvinalias-portfolio-ml-api.hf.space/industrial/health
curl -s https://alvinalias-portfolio-ml-api.hf.space/cmapss/health
curl -s https://alvinalias-portfolio-ml-api.hf.space/hvac/health
curl -s https://alvinalias-portfolio-ml-api.hf.space/maintenance/health
```

First request per prefix may take 30–90s (cold start + model load). Retry once.

If `/hvac/health` fails on sklearn version, note the error; HVAC artifacts were trained on sklearn 1.7.2 and may need a pin adjustment in `requirements.txt`.

## 6. Push frontend cutover (five repos)

Frontend files in this workspace already point at `https://alvinalias-portfolio-ml-api.hf.space/<prefix>`. Push each project's frontend to GitHub/Vercel **only after** step 5 passes:

- `retail_returns_intelligence`
- `industrial_failure_classification`
- `cmapss_rul`
- `hvac_equipment_health`
- `maintenance_nlp`
- `portfolio_site` (playground API URLs)

## 7. Suspend Render (five APIs only)

After frontends are live on HF:

- Suspend Render services for retail, industrial, cmapss, hvac, maintenance
- **Keep** RAG on Render

## 8. Re-sync after project changes

When you change API code or models in a source project:

```bash
bash scripts/sync_from_portfolio.sh
git add services/
git commit -m "Sync services from portfolio projects"
git push
```

GitHub Actions pushes to HF automatically.

## Manual HF push (optional)

If Actions is not set up yet:

```bash
bash scripts/sync_from_portfolio.sh
# clone HF Space to ./hf-space, rsync repo contents, commit, push
```

See `PUSH_TO_HF.md` for the rsync pattern.
