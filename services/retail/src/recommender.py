"""
src/recommender.py - Substitute Product Recommender (Model 4).

Hybrid approach:
  - Content track: sentence-transformers embeddings on product Description text
  - CF track: implicit ALS on customer × product interaction matrix
  - Blend: 0.5 * content_rank + 0.5 * cf_rank (weighted rank fusion)

Evaluation: Recall@K, MRR, NDCG@10 on held-out next-purchase as ground truth.
Business metric: % of recommended substitutes NOT in the customer's return history.

Full implementation in notebooks/10_substitute_recommender.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from implicit import als


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Content-based: product description embeddings
# ---------------------------------------------------------------------------

def build_description_embeddings(
    df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> tuple[np.ndarray, pd.Series]:
    """Encode unique product descriptions into a dense embedding matrix.

    Returns:
        embeddings: (n_products, embedding_dim) float32 array
        stock_codes: Series of stock_code values aligned with embedding rows
    """
    product_df = (
        df.dropna(subset=["description", "stock_code"])
        .groupby("stock_code")["description"]
        .first()
        .reset_index()
    )
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        product_df["description"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32), product_df["stock_code"]


def content_similarity(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Return indices of top-k most similar products by cosine similarity.

    Embeddings must be L2-normalized (normalize_embeddings=True in encode).
    """
    scores = corpus_embeddings @ query_embedding
    return np.argsort(scores)[::-1][:top_k]


# ---------------------------------------------------------------------------
# Collaborative filtering: implicit ALS
# ---------------------------------------------------------------------------

def build_interaction_matrix(
    df: pd.DataFrame,
) -> tuple[sp.csr_matrix, pd.Series, pd.Series]:
    """Build a customer × product implicit interaction matrix.

    Interaction strength = number of purchases (not returns).
    Returns: (matrix, customer_index, product_index)
    """
    purchases = df[(df["is_return"] == 0) & df["customer_id"].notna()].copy()
    cust_idx = pd.CategoricalIndex(purchases["customer_id"].unique())
    prod_idx = pd.CategoricalIndex(purchases["stock_code"].unique())

    row = pd.Categorical(purchases["customer_id"], categories=cust_idx).codes
    col = pd.Categorical(purchases["stock_code"], categories=prod_idx).codes
    data = np.ones(len(purchases))

    matrix = sp.csr_matrix(
        (data, (row, col)), shape=(len(cust_idx), len(prod_idx))
    )
    return matrix, cust_idx, prod_idx


def train_als(
    matrix: sp.csr_matrix,
    factors: int = 64,
    iterations: int = 20,
    regularization: float = 0.01,
    random_state: int = 42,
) -> als.AlternatingLeastSquares:
    """Train ALS model from the implicit library."""
    model = als.AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state,
    )
    model.fit(matrix)
    return model


# ---------------------------------------------------------------------------
# Hybrid rank fusion
# ---------------------------------------------------------------------------

def hybrid_recommend(
    content_ranks: np.ndarray,
    cf_ranks: np.ndarray,
    alpha: float = 0.5,
    top_k: int = 3,
) -> np.ndarray:
    """Weighted rank fusion: alpha * content_rank + (1-alpha) * cf_rank.

    Lower combined rank = better recommendation.
    Returns indices of top_k products.
    """
    combined = alpha * content_ranks + (1 - alpha) * cf_ranks
    return np.argsort(combined)[:top_k]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def recall_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """Recall@K: fraction of relevant items found in top-k recommendations."""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def reciprocal_rank(recommended: list[str], relevant: list[str]) -> float:
    """Mean Reciprocal Rank component: 1/rank of first relevant item."""
    for i, item in enumerate(recommended, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(recommended: list[str], relevant: list[str], k: int) -> float:
    """NDCG@K: normalized discounted cumulative gain."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    k: int = 10,
) -> dict[str, float]:
    """Compute Recall@K, MRR, and NDCG@K across all queries.

    Args:
        recommendations: {customer_id: [stock_code, ...]} top-k suggestions
        ground_truth: {customer_id: [stock_code, ...]} held-out next purchases
    """
    recalls, rrs, ndcgs = [], [], []
    for cid, recs in recommendations.items():
        relevant = ground_truth.get(cid, [])
        recalls.append(recall_at_k(recs, relevant, k))
        rrs.append(reciprocal_rank(recs, relevant))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
    return {
        f"recall@{k}": float(np.mean(recalls)),
        "mrr": float(np.mean(rrs)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_recommender_artifacts(
    embeddings: np.ndarray,
    stock_codes: pd.Series,
    als_model: als.AlternatingLeastSquares,
    cust_idx: pd.CategoricalIndex,
    prod_idx: pd.CategoricalIndex,
) -> None:
    """Persist all recommender artifacts to models/."""
    MODELS_DIR.mkdir(exist_ok=True)
    np.save(MODELS_DIR / "product_embeddings.npy", embeddings)
    joblib.dump(stock_codes, MODELS_DIR / "embedding_stock_codes.joblib")
    joblib.dump(als_model, MODELS_DIR / "als_model.joblib")
    joblib.dump(cust_idx, MODELS_DIR / "als_customer_index.joblib")
    joblib.dump(prod_idx, MODELS_DIR / "als_product_index.joblib")
