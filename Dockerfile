FROM python:3.13-slim

WORKDIR /sentiment_app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files (dataset is embedded inside train.py)
COPY train.py .
COPY app.py .

# Train the model at build time
# MLflow logs → /sentiment_app/mlruns
# Trained model → /sentiment_app/models/sentiment_model.pkl
RUN python train.py

# Railway injects $PORT at runtime; fallback to 8000 locally
ENV PORT=8000
ENV MODEL_PATH=models/sentiment_model.pkl
ENV MLFLOW_TRACKING_URI=mlruns

EXPOSE 8000

CMD ["python", "-c", "import os,uvicorn; uvicorn.run('app:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))"]
