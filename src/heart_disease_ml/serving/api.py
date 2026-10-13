"""FastAPI application for Heart Disease inference and monitoring."""

import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "champion" / "model.pkl"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------

REQUEST_COUNT = Counter(
    "heart_disease_api_requests",
    "Total HTTP requests handled by the Heart Disease API.",
    labelnames=("method", "endpoint", "status_code"),
)

REQUEST_DURATION = Histogram(
    "heart_disease_api_request_duration_seconds",
    "HTTP request duration in seconds.",
    labelnames=("method", "endpoint"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUESTS_IN_PROGRESS = Gauge(
    "heart_disease_api_requests_in_progress",
    "Number of HTTP requests currently being processed.",
)

# ------------------------------------------------------------------
# Load the existing Section 4 champion model
# ------------------------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Champion model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if not hasattr(model, "predict_proba"):
    raise TypeError("The loaded model must support predict_proba().")

logger.info("Champion model loaded from %s", MODEL_PATH)

# ------------------------------------------------------------------
# FastAPI application and request instrumentation
# ------------------------------------------------------------------

app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Predict heart disease presence using the packaged "
        "scikit-learn pipeline."
    ),
    version="1.1.0",
)


@app.middleware("http")
async def instrument_requests(request: Request, call_next):
    """Record request count, latency, in-progress count and access logs."""

    # Do not count or log Prometheus scrapes as application traffic.
    if request.url.path == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = 500
    REQUESTS_IN_PROGRESS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        logger.exception(
            "Request failed: method=%s path=%s",
            request.method,
            request.url.path,
        )
        raise
    finally:
        duration = time.perf_counter() - start_time

        # Use the route template (e.g. /predict) instead of a raw URL
        # to avoid creating a separate metric for every URL variation.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)

        REQUEST_COUNT.labels(
            request.method,
            endpoint,
            str(status_code),
        ).inc()

        REQUEST_DURATION.labels(
            request.method,
            endpoint,
        ).observe(duration)

        REQUESTS_IN_PROGRESS.dec()

        logger.info(
            "HTTP request method=%s endpoint=%s status_code=%s "
            "duration_seconds=%.4f",
            request.method,
            endpoint,
            status_code,
            duration,
        )


# ------------------------------------------------------------------
# Request schema
# ------------------------------------------------------------------

class HeartDiseaseFeatures(BaseModel):
    """Input schema containing the 13 model features."""

    model_config = ConfigDict(extra="forbid")

    age: float | None
    sex: float | None
    cp: float | None
    trestbps: float | None
    chol: float | None
    fbs: float | None
    restecg: float | None
    thalach: float | None
    exang: float | None
    oldpeak: float | None
    slope: float | None
    ca: float | None
    thal: float | None


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------

@app.get("/health")
def health_check() -> dict:
    """Report whether the API and model are available."""

    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


# ------------------------------------------------------------------
# Prometheus scrape endpoint
# ------------------------------------------------------------------

@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose application metrics in Prometheus text format."""

    return Response(
        content=generate_latest(),
        headers={"Content-Type": CONTENT_TYPE_LATEST},
    )


# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------

@app.post("/predict")
def predict(features: HeartDiseaseFeatures) -> dict:
    """Return the predicted class and its probability."""

    input_data = pd.DataFrame(
        [features.model_dump()],
        columns=FEATURE_COLUMNS,
    )

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    class_probabilities = {
        int(class_label): float(probability)
        for class_label, probability in zip(
            model.classes_,
            probabilities,
        )
    }

    confidence = class_probabilities[prediction]

    logger.info(
        "Prediction completed: class=%d, confidence=%.4f",
        prediction,
        confidence,
    )

    return {
        "prediction": prediction,
        "prediction_label": (
            "heart_disease_present"
            if prediction == 1
            else "no_heart_disease_detected"
        ),
        "confidence": confidence,
        "class_probabilities": {
            "no_heart_disease": class_probabilities[0],
            "heart_disease_present": class_probabilities[1],
        },
    }
