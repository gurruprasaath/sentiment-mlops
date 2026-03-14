"""
FastAPI application for Sentiment Analysis
Serves predictions from the trained Scikit-learn pipeline.
"""

import os
import json
import time
import joblib
import numpy as np
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/sentiment_model.pkl")
METRICS_PATH = "models/metrics.json"

model = None
model_metrics: dict = {}


def load_model():
    global model, model_metrics
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python train.py' first to train the model."
        )
    model = joblib.load(MODEL_PATH)
    if Path(METRICS_PATH).exists():
        with open(METRICS_PATH) as f:
            model_metrics = json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    print("Model loaded successfully.")
    yield


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "A Sentiment Analysis API built with **Scikit-learn** (TF-IDF + Logistic Regression), "
        "tracked with **MLflow**, containerised with **Docker**, and deployed on **Railway**.\n\n"
        "### Endpoints\n"
        "- `POST /predict` – Predict sentiment for a single text\n"
        "- `POST /predict/batch` – Predict sentiment for multiple texts\n"
        "- `GET /health` – Health check\n"
        "- `GET /metrics` – Model evaluation metrics\n"
        "- `GET /` – Interactive demo UI\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="This movie was absolutely amazing! I loved every scene.",
    )


class PredictResponse(BaseModel):
    text: str
    sentiment: str  # "positive" | "negative"
    label: int  # 1 | 0
    confidence: float  # probability of predicted class
    probabilities: dict  # {"positive": float, "negative": float}
    inference_time_ms: float


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_texts: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    version: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def make_prediction(text: str) -> PredictResponse:
    start = time.perf_counter()
    proba = model.predict_proba([text])[0]  # [neg_prob, pos_prob]
    label = int(np.argmax(proba))
    sentiment = "positive" if label == 1 else "negative"
    confidence = float(proba[label])
    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        text=text,
        sentiment=sentiment,
        label=label,
        confidence=round(confidence, 4),
        probabilities={
            "positive": round(float(proba[1]), 4),
            "negative": round(float(proba[0]), 4),
        },
        inference_time_ms=round(elapsed_ms, 3),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def root():
    """Interactive demo page."""
    return HTMLResponse(content=DEMO_HTML)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        version="1.0.0",
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Return training evaluation metrics."""
    return {
        "model": "TF-IDF + Logistic Regression",
        "dataset": "Custom Sentiment Dataset",
        "metrics": model_metrics,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict the sentiment of a single text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return make_prediction(request.text)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """Predict the sentiment of multiple texts at once."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start = time.perf_counter()
    results = [make_prediction(t) for t in request.texts]
    total_ms = (time.perf_counter() - start) * 1000

    return BatchPredictResponse(
        results=results,
        total_texts=len(results),
        inference_time_ms=round(total_ms, 3),
    )


# ---------------------------------------------------------------------------
# Embedded demo UI
# ---------------------------------------------------------------------------
DEMO_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Sentiment Analysis API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e0e0f0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: flex-start; padding: 40px 16px; }
    h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px; }
    .subtitle { color: #94a3b8; margin-bottom: 36px; font-size: 1rem; }
    .card { background: #1a1a2e; border: 1px solid #2d2d4e; border-radius: 16px; padding: 32px; width: 100%; max-width: 700px; margin-bottom: 24px; }
    label { display: block; font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
    textarea { width: 100%; height: 110px; background: #0f0f1a; border: 1px solid #3d3d6e; border-radius: 10px; color: #e0e0f0; padding: 12px 14px; font-size: 1rem; resize: vertical; outline: none; transition: border-color 0.2s; }
    textarea:focus { border-color: #a78bfa; }
    button { margin-top: 16px; width: 100%; padding: 14px; background: linear-gradient(90deg, #7c3aed, #2563eb); border: none; border-radius: 10px; color: #fff; font-size: 1.05rem; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
    button:hover { opacity: 0.88; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .result { margin-top: 24px; display: none; }
    .badge { display: inline-block; padding: 6px 18px; border-radius: 999px; font-weight: 700; font-size: 1.1rem; margin-bottom: 12px; }
    .positive { background: #14532d; color: #4ade80; }
    .negative { background: #450a0a; color: #f87171; }
    .meta { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; }
    .meta-item { background: #0f0f1a; border: 1px solid #2d2d4e; border-radius: 8px; padding: 10px 16px; flex: 1; min-width: 140px; }
    .meta-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; }
    .meta-value { font-size: 1.15rem; font-weight: 600; color: #a78bfa; margin-top: 4px; }
    .links { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin-top: 8px; }
    .links a { color: #60a5fa; text-decoration: none; font-size: 0.9rem; border: 1px solid #2d2d4e; padding: 6px 14px; border-radius: 8px; transition: background 0.2s; }
    .links a:hover { background: #1a1a2e; }
    .tech { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin-bottom: 24px; }
    .tag { background: #1e1e3f; border: 1px solid #3d3d6e; border-radius: 999px; padding: 4px 14px; font-size: 0.8rem; color: #94a3b8; }
    .error { color: #f87171; margin-top: 12px; font-size: 0.9rem; display: none; }
    .bar-wrap { margin-top: 14px; }
    .bar-label { font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px; display: flex; justify-content: space-between; }
    .bar-bg { background: #0f0f1a; border-radius: 999px; height: 10px; overflow: hidden; border: 1px solid #2d2d4e; }
    .bar-fill-pos { background: linear-gradient(90deg, #22c55e, #4ade80); height: 100%; border-radius: 999px; transition: width 0.5s ease; }
    .bar-fill-neg { background: linear-gradient(90deg, #ef4444, #f87171); height: 100%; border-radius: 999px; transition: width 0.5s ease; }
  </style>
</head>
<body>
  <h1>Sentiment Analysis</h1>
  <p class="subtitle">MLOps Demo &mdash; Scikit-learn &bull; MLflow &bull; Docker &bull; Railway</p>
  <div class="tech">
    <span class="tag">Scikit-learn</span>
    <span class="tag">TF-IDF</span>
    <span class="tag">Logistic Regression</span>
    <span class="tag">MLflow</span>
    <span class="tag">FastAPI</span>
    <span class="tag">Docker</span>
    <span class="tag">Railway</span>
  </div>

  <div class="card">
    <label for="reviewText">Enter text to analyse</label>
    <textarea id="reviewText" placeholder="e.g. This movie was absolutely fantastic! I loved every minute of it."></textarea>
    <button id="predictBtn" onclick="predict()">Analyse Sentiment</button>
    <p class="error" id="errorMsg"></p>

    <div class="result" id="result">
      <div class="badge" id="sentimentBadge"></div>
      <div class="bar-wrap">
        <div class="bar-label"><span>Positive</span><span id="posVal"></span></div>
        <div class="bar-bg"><div class="bar-fill-pos" id="posBar" style="width:0%"></div></div>
      </div>
      <div class="bar-wrap" style="margin-top:8px">
        <div class="bar-label"><span>Negative</span><span id="negVal"></span></div>
        <div class="bar-bg"><div class="bar-fill-neg" id="negBar" style="width:0%"></div></div>
      </div>
      <div class="meta">
        <div class="meta-item"><div class="meta-label">Confidence</div><div class="meta-value" id="confidence"></div></div>
        <div class="meta-item"><div class="meta-label">Inference Time</div><div class="meta-value" id="infTime"></div></div>
      </div>
    </div>
  </div>

  <div class="links">
    <a href="/docs" target="_blank">Swagger UI</a>
    <a href="/redoc" target="_blank">ReDoc</a>
    <a href="/health" target="_blank">Health Check</a>
    <a href="/metrics" target="_blank">Model Metrics</a>
  </div>

  <script>
    async function predict() {
      const text = document.getElementById('reviewText').value.trim();
      const btn = document.getElementById('predictBtn');
      const errEl = document.getElementById('errorMsg');
      errEl.style.display = 'none';
      if (!text) { errEl.textContent = 'Please enter some text.'; errEl.style.display = 'block'; return; }
      btn.disabled = true; btn.textContent = 'Analysing...';
      try {
        const res = await fetch('/predict', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({text}) });
        if (!res.ok) throw new Error((await res.json()).detail || 'Error');
        const d = await res.json();
        const badge = document.getElementById('sentimentBadge');
        badge.textContent = d.sentiment.toUpperCase();
        badge.className = 'badge ' + d.sentiment;
        document.getElementById('posVal').textContent = (d.probabilities.positive * 100).toFixed(1) + '%';
        document.getElementById('negVal').textContent = (d.probabilities.negative * 100).toFixed(1) + '%';
        document.getElementById('posBar').style.width = (d.probabilities.positive * 100) + '%';
        document.getElementById('negBar').style.width = (d.probabilities.negative * 100) + '%';
        document.getElementById('confidence').textContent = (d.confidence * 100).toFixed(2) + '%';
        document.getElementById('infTime').textContent = d.inference_time_ms.toFixed(2) + ' ms';
        document.getElementById('result').style.display = 'block';
      } catch(e) { errEl.textContent = e.message; errEl.style.display = 'block'; }
      finally { btn.disabled = false; btn.textContent = 'Analyse Sentiment'; }
    }
    document.getElementById('reviewText').addEventListener('keydown', e => { if (e.key === 'Enter' && e.ctrlKey) predict(); });
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Entry point (for local dev)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False
    )
