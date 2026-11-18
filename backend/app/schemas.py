"""
Pydantic Schemas
================
Request and response models for the Fake News Detection API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description="Headline / title of the news article.",
        example="Scientists Discover Cure for All Diseases",
    )
    text: str = Field(
        ...,
        min_length=10,
        description="Full body text of the article to classify.",
        example="Researchers at a leading university have announced a breakthrough ...",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Breaking: Shocking Health Miracle Revealed",
                "text": "According to anonymous sources, a secret formula has been suppressed by Big Pharma for decades. Click here to learn the truth they don't want you to know!",
            }
        }


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted class: 'REAL' or 'FAKE'.")
    confidence: float = Field(..., description="Probability of the predicted class (0–1).")
    probabilities: dict[str, float] = Field(
        ..., description="Softmax probabilities for each class."
    )
    explanation: str = Field(..., description="Human-readable rationale for the prediction.")
    latency_ms: float = Field(..., description="Server-side inference time in milliseconds.")

    class Config:
        json_schema_extra = {
            "example": {
                "label": "FAKE",
                "confidence": 0.9412,
                "probabilities": {"REAL": 0.0588, "FAKE": 0.9412},
                "explanation": "The model classified this article as FAKE with high confidence (94.1%). ...",
                "latency_ms": 48.3,
            }
        }


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when the service is healthy.")
    model_loaded: bool = Field(..., description="True when the model is fully loaded.")
    timestamp: float = Field(..., description="Unix timestamp of the health check.")


class ModelInfoResponse(BaseModel):
    model_name: str
    num_labels: int
    max_length: int
    accuracy: float = Field(..., description="Validation accuracy on LIAR dataset.")
    training_samples: int
    dataset: str
    labels: list[str]
