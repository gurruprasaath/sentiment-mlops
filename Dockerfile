# ── Stage 1: Train the model ───────────────────────────────────────────────
FROM python:3.13-slim AS trainer

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY train.py .
COPY data/ ./data/

# Train the model (logs stored in /app/mlruns, model in /app/models)
RUN python train.py

# ── Stage 2: Serve the API ─────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY app.py .

# Copy trained artefacts from trainer stage
COPY --from=trainer /app/models ./models
COPY --from=trainer /app/mlruns ./mlruns

# Railway injects $PORT at runtime; default to 8000 for local use
ENV PORT=8000
ENV MODEL_PATH=models/sentiment_model.pkl
ENV MLFLOW_TRACKING_URI=mlruns

# Expose the port
EXPOSE ${PORT}

# Health-check so Railway knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# Start the API; Railway will set $PORT automatically
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
