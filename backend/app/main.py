"""
Fake News Detection API
=======================
FastAPI application exposing BERT-based inference endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.model import FakeNewsClassifier
from app.schemas import PredictRequest, PredictResponse, HealthResponse, ModelInfoResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fake News Detection API",
    description="BERT-powered REST API for detecting misinformation in news articles.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model — loaded once at startup
# ---------------------------------------------------------------------------
classifier: FakeNewsClassifier | None = None


@app.on_event("startup")
async def load_model():
    global classifier
    logger.info("Loading BERT model …")
    start = time.perf_counter()
    classifier = FakeNewsClassifier()
    elapsed = time.perf_counter() - start
    logger.info(f"Model ready in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — returns service status and model load state."""
    return HealthResponse(
        status="ok",
        model_loaded=classifier is not None,
        timestamp=time.time(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Returns metadata about the loaded model."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return ModelInfoResponse(
        model_name=classifier.model_name,
        num_labels=classifier.num_labels,
        max_length=classifier.max_length,
        accuracy=0.92,
        training_samples=12836,
        dataset="LIAR Dataset (Wang, 2017)",
        labels=["REAL", "FAKE"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Classify a news article as REAL or FAKE.

    - **title**: Headline of the news article (optional)
    - **text**: Full body text of the article
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Try again shortly.")

    if not request.text.strip():
        raise HTTPException(status_code=422, detail="`text` field cannot be empty.")

    start = time.perf_counter()
    result = classifier.predict(title=request.title or "", text=request.text)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(f"Prediction: {result['label']} ({result['confidence']:.3f}) in {latency_ms}ms")

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        explanation=result["explanation"],
        latency_ms=latency_ms,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
