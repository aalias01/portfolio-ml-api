# Manual push to Hugging Face Space

Prefer GitHub Actions (`.github/workflows/sync_hf_space.yml`) after `HF_TOKEN` is set.

## One-time local push

```bash
cd "$(dirname "$0")/.."
bash scripts/sync_from_portfolio.sh

export HF_TOKEN="hf_..."   # write token

HF_TOKEN_ENCODED="$(python3 -c "import os, urllib.parse; print(urllib.parse.quote(os.environ['HF_TOKEN'], safe=''))")"

rm -rf hf-space
GIT_LFS_SKIP_SMUDGE=1 git clone \
  "https://alvinalias:${HF_TOKEN_ENCODED}@huggingface.co/spaces/alvinalias/portfolio-ml-api" \
  hf-space

rsync -av ./ hf-space/ \
  --exclude .git \
  --exclude hf-space \
  --exclude docs_local \
  --exclude YOUR_STEPS.md \
  --exclude ARCHITECTURE.md

cd hf-space
git add README.md Dockerfile requirements.txt .gitignore gateway services
git commit -m "Manual sync from local portfolio_ml_api"
git push
```

Space URL: https://alvinalias-portfolio-ml-api.hf.space
