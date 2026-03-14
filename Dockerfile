FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY train.py .
COPY app.py .

# Train the model at build time (MLflow logs to /app/mlruns, model saved to /app/models)
RUN python train.py

# Railway sets $PORT automatically; default 8000 for local use
ENV PORT=8000
ENV MODEL_PATH=models/sentiment_model.pkl
ENV MLFLOW_TRACKING_URI=mlruns

EXPOSE 8000

# Start FastAPI
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
