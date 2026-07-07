from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    air_temperature_k: float    = Field(..., description="Air temperature [K]", ge=290, le=310, examples=[298.1])
    process_temperature_k: float = Field(..., description="Process temperature [K]", ge=305, le=315, examples=[308.6])
    rotational_speed_rpm: float = Field(..., description="Rotational speed [rpm]", ge=1168, le=2886, examples=[1551])
    torque_nm: float            = Field(..., description="Torque [Nm]", ge=3.8, le=76.6, examples=[42.8])
    tool_wear_min: float        = Field(..., description="Tool wear [min]", ge=0, le=253, examples=[0])

    model_config = {
        "json_schema_extra": {"example": {
            "air_temperature_k": 298.1,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1551,
            "torque_nm": 42.8,
            "tool_wear_min": 0,
        }}
    }


class SHAPFactor(BaseModel):
    feature: str
    shap_value: float
    direction: str
    feature_value: float


class PredictResponse(BaseModel):
    failure_probability: float  = Field(..., ge=0, le=1)
    risk_level: str             = Field(..., examples=["low", "medium", "high"])
    prediction: int             = Field(..., description="1=failure predicted at optimal threshold")
    threshold_used: float
    estimated_cost_if_ignored: Optional[int] = Field(default=None, description="FN cost if failure missed")
    top_shap_factors: Optional[list[SHAPFactor]] = None


class RiskZones(BaseModel):
    low_below: float
    high_at: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    optimal_threshold: float
    risk_zones: RiskZones
    version: str = "0.1.0"
