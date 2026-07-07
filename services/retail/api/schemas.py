"""
api/schemas.py - Pydantic request/response models for Retail Returns Intelligence API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /score: score a single transaction
# ---------------------------------------------------------------------------

class TransactionScoreRequest(BaseModel):
    customer_id: str = Field(..., description="CustomerID from UCI dataset")
    invoice_no: str = Field(..., description="InvoiceNo of the transaction to score")
    stock_code: str = Field(..., description="StockCode of the primary item")
    quantity: float = Field(..., ge=1, description="Quantity ordered")
    unit_price: float = Field(..., ge=0, description="Unit price in GBP")
    country: str = Field(default="United Kingdom", description="Country of customer")
    unit_price_z: float | None = Field(default=None, description="Optional real-invoice unit price z-score")
    quantity_z: float | None = Field(default=None, description="Optional real-invoice quantity z-score")
    is_weekend: int | None = Field(default=None, ge=0, le=1, description="Optional real-invoice weekend flag")
    month_end_proximity: int | None = Field(default=None, ge=0, description="Optional real-invoice days from month end")
    category_return_rate: float | None = Field(default=None, ge=0.0, le=1.0, description="Optional stock-level return rate")

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "12345",
            "invoice_no": "536365",
            "stock_code": "85123A",
            "quantity": 6,
            "unit_price": 2.55,
            "country": "United Kingdom",
        }
    }}


class ShapEntry(BaseModel):
    feature: str
    value: float
    direction: str = Field(..., description="'increases' or 'decreases' return risk")


class TransactionScoreResponse(BaseModel):
    customer_id: str
    invoice_no: str
    return_probability: float = Field(..., ge=0.0, le=1.0)
    risk_tier: str = Field(..., description="High / Medium / Low based on threshold")
    segment: str = Field(..., description="Customer segment: Premium Loyal / Healthy Browser / At-Risk / Returner")
    anomaly_flag: int = Field(..., description="1 = customer flagged by the behavior anomaly detector")
    anomaly_score: float = Field(..., description="Isolation Forest decision score (lower = more anomalous)")
    top_shap_factors: list[ShapEntry]


# ---------------------------------------------------------------------------
# /demo-cases: curated real invoice examples for the frontend
# ---------------------------------------------------------------------------

class DemoFilterOption(BaseModel):
    key: str
    label: str


class DemoCase(BaseModel):
    case_id: str
    invoice_no: str
    customer_id: str
    stock_code: str
    description: str
    quantity: float
    unit_price: float
    country: str
    invoice_date: str
    segment: str
    risk_tier: str
    return_probability: float = Field(..., ge=0.0, le=1.0)
    anomaly_flag: int
    anomaly_score: float
    lifetime_return_rate: float
    frequency_score: int
    monetary_score: float
    has_substitutes: bool
    unit_price_z: float
    quantity_z: float
    is_weekend: int
    month_end_proximity: int
    category_return_rate: float
    tags: list[str]


class DemoCasesResponse(BaseModel):
    filters: list[DemoFilterOption]
    cases: list[DemoCase]


# ---------------------------------------------------------------------------
# /customer/{customer_id}/profile: full behavioral profile
# ---------------------------------------------------------------------------

class CustomerProfileResponse(BaseModel):
    customer_id: str
    segment: str
    anomaly_flag: int
    anomaly_score: float
    lifetime_return_rate: float
    return_value_ratio: float
    return_velocity: float = Field(..., description="Returns in last 30 days")
    tenure_days: int
    recency_score: int = Field(..., description="Days since last purchase")
    frequency_score: int = Field(..., description="Total orders")
    monetary_score: float = Field(..., description="Lifetime revenue in GBP")
    top_shap_factors: list[ShapEntry]


# ---------------------------------------------------------------------------
# /substitutes/{invoice_no}: substitute product recommendations
# ---------------------------------------------------------------------------

class SubstituteItem(BaseModel):
    stock_code: str
    description: str
    content_similarity: float = Field(..., ge=0.0, le=1.0)
    in_customer_return_history: bool = Field(..., description="True if customer has returned this item before")
    rationale: str = Field(..., description="Brief explanation of why this substitute was recommended")


class SubstitutesResponse(BaseModel):
    invoice_no: str
    original_stock_code: str
    original_description: str
    substitutes: list[SubstituteItem] = Field(..., max_length=3)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    risk_tiers: dict[str, float] | None = None
    version: str = "1.0.0"
