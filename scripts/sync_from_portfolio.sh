#!/usr/bin/env bash
# Copy serving subsets from the five portfolio project repos into services/.
# Run from anywhere: bash scripts/sync_from_portfolio.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="$(cd "${REPO_ROOT}/.." && pwd)"

patch_service_paths() {
  local dest="$1"
  while IFS= read -r -d '' file; do
    perl -i -pe 's/^(MODELS_DIR|MODEL_DIR) = Path\("models"\)/$1 = Path(__file__).resolve().parent.parent \/ "models"/' "${file}"
    perl -i -pe 's/^DATA_DIR = Path\("data"\)/DATA_DIR = Path(__file__).resolve().parent.parent \/ "data"/' "${file}"
  done < <(find "${dest}" -name '*.py' -print0)
}

copy_models() {
  local src_root="$1"
  local dest_models="$2"
  shift 2
  local files=("$@")

  mkdir -p "${dest_models}"
  for file in "${files[@]}"; do
    local src="${src_root}/models/${file}"
    if [[ ! -f "${src}" && -f "${src_root}/deploy/hf_space/models/${file}" ]]; then
      src="${src_root}/deploy/hf_space/models/${file}"
    fi
    if [[ -f "${src}" ]]; then
      cp "${src}" "${dest_models}/${file}"
    else
      echo "WARNING: missing artifact ${file} for ${src_root}" >&2
    fi
  done
}

sync_code() {
  local src_root="$1"
  local dest="$2"
  local with_src="${3:-1}"

  rm -rf "${dest}/api"
  [[ "${with_src}" == "1" ]] && rm -rf "${dest}/src"
  mkdir -p "${dest}"

  rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
    "${src_root}/api/" "${dest}/api/"
  if [[ "${with_src}" == "1" && -d "${src_root}/src" ]]; then
    rsync -a --delete --exclude '__pycache__' --exclude '*.pyc' \
      "${src_root}/src/" "${dest}/src/"
  fi
  patch_service_paths "${dest}"
}

echo "Workspace: ${WORKSPACE}"
echo "Repo root: ${REPO_ROOT}"

# --- retail ---
RETAIL="${WORKSPACE}/retail_returns_intelligence"
RETAIL_DEST="${REPO_ROOT}/services/retail"
sync_code "${RETAIL}" "${RETAIL_DEST}" 1
copy_models "${RETAIL}" "${RETAIL_DEST}/models" \
  classifier.joblib anomaly_detector.joblib anomaly_scaler.joblib \
  segmentation_kmeans.joblib segmentation_scaler.joblib customer_features.joblib \
  invoice_substitutes.joblib demo_cases.joblib MODEL_CARD.md classifier_meta.json
for opt in product_embeddings.npy embedding_stock_codes.joblib als_model.joblib \
  als_product_index.joblib als_customer_index.joblib; do
  src="${RETAIL}/models/${opt}"
  [[ -f "${src}" ]] && cp "${src}" "${RETAIL_DEST}/models/${opt}" || true
done

# --- industrial ---
IND="${WORKSPACE}/industrial_failure_classification"
IND_DEST="${REPO_ROOT}/services/industrial"
sync_code "${IND}" "${IND_DEST}" 1
copy_models "${IND}" "${IND_DEST}/models" \
  xgb_classifier.joblib scaler.joblib model_meta.json MODEL_CARD.md

# --- cmapss ---
CMAPSS="${WORKSPACE}/cmapss_rul"
CMAPSS_DEST="${REPO_ROOT}/services/cmapss"
sync_code "${CMAPSS}" "${CMAPSS_DEST}" 0
copy_models "${CMAPSS}" "${CMAPSS_DEST}/models" xgb_rul.joblib MODEL_CARD.md

# --- hvac ---
HVAC="${WORKSPACE}/hvac_equipment_health"
HVAC_DEST="${REPO_ROOT}/services/hvac"
sync_code "${HVAC}" "${HVAC_DEST}" 1
copy_models "${HVAC}" "${HVAC_DEST}/models" \
  isolation_forest.joblib isolation_forest_scaler.joblib lof_model.joblib \
  scorer_meta.json unit_baselines.joblib unit_baselines_meta.json \
  demo_readings.json MODEL_CARD.md

# --- maintenance ---
MAINT="${WORKSPACE}/maintenance_nlp"
MAINT_DEST="${REPO_ROOT}/services/maintenance"
sync_code "${MAINT}" "${MAINT_DEST}" 1
copy_models "${MAINT}" "${MAINT_DEST}/models" \
  tfidf_pipeline.joblib corpus_meta.json embeddings_texts.json MODEL_CARD.md
if [[ -f "${MAINT}/models/embeddings_index.npy" ]]; then
  cp "${MAINT}/models/embeddings_index.npy" "${MAINT_DEST}/models/embeddings_index.npy"
fi
if [[ -d "${MAINT}/models/onnx" ]]; then
  mkdir -p "${MAINT_DEST}/models/onnx"
  rsync -a \
    --exclude 'model.onnx' \
    --exclude '*.onnx.data' \
    --exclude '*.onnx_data' \
    "${MAINT}/models/onnx/" "${MAINT_DEST}/models/onnx/"
fi

echo ""
echo "Sync complete. Service tree:"
find "${REPO_ROOT}/services" -maxdepth 3 -type f | sort
