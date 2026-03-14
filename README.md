# Sentiment Analysis — MLOps Project

A production-grade **Sentiment Analysis** API demonstrating a complete MLOps workflow:

| Technology | Purpose |
|---|---|
| **Scikit-learn** | TF-IDF + Logistic Regression model |
| **MLflow** | Experiment tracking & model registry |
| **FastAPI** | REST API for serving predictions |
| **Docker** | Multi-stage containerisation |
| **Railway** | Cloud deployment |

---

## Project Structure

```
sentiment-mlops/
├── train.py            # ML training script (MLflow tracked)
├── app.py              # FastAPI prediction API
├── requirements.txt    # Python dependencies
├── Dockerfile          # Multi-stage Docker build
├── railway.toml        # Railway deployment config
├── .env.example        # Environment variable template
├── .gitignore
├── models/             # Generated at train time (git-ignored)
│   ├── sentiment_model.pkl
│   └── metrics.json
└── mlruns/             # MLflow tracking data (git-ignored)
```

---

## Quick Start (Local)

### 1. Clone & install dependencies

```bash
git clone <your-repo-url>
cd sentiment-mlops
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

MLflow logs experiments to `./mlruns/`. View the UI:

```bash
mlflow ui
# Open http://localhost:5000
```

### 3. Start the API

```bash
python app.py
# or
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** for the interactive demo UI.
Open **http://localhost:8000/docs** for the Swagger API docs.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive demo UI |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Model evaluation metrics |
| `POST` | `/predict` | Single text prediction |
| `POST` | `/predict/batch` | Batch prediction |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

### Example — Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic!"}'
```

Response:
```json
{
  "text": "This movie was absolutely fantastic!",
  "sentiment": "positive",
  "label": 1,
  "confidence": 0.9823,
  "probabilities": { "positive": 0.9823, "negative": 0.0177 },
  "inference_time_ms": 1.234
}
```

### Example — Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great movie!", "Terrible film, waste of time."]}'
```

---

## Docker

### Build & run locally

```bash
# Build (trains model inside the image)
docker build -t sentiment-mlops .

# Run
docker run -p 8000:8000 sentiment-mlops

# Open http://localhost:8000
```

The **multi-stage Dockerfile**:
1. **Stage 1 (trainer)** — Installs deps, runs `train.py`, produces `models/`
2. **Stage 2 (runtime)** — Copies only the trained model + API code → lean final image

---

## Deploy to Railway

### Option A — GitHub (recommended)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Select your repo — Railway auto-detects the `Dockerfile`
4. Click **Deploy** — Railway will build the image and give you a public URL

### Option B — Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Create project and deploy
railway init
railway up
```

Railway automatically:
- Injects `$PORT` into the container
- Provides HTTPS URL
- Restarts on failure (configured in `railway.toml`)

---

## MLflow Experiment Tracking

The training script logs:

- **Parameters**: `max_features`, `ngram_range`, `C`, `max_iter`, `test_size`
- **Metrics**: `accuracy`, `precision`, `recall`, `f1_score`
- **Artefacts**: trained sklearn pipeline (registered in Model Registry)

View locally:
```bash
mlflow ui
```

---

## Model Details

| Component | Value |
|---|---|
| Vectoriser | TF-IDF (`max_features=5000`, `ngram_range=(1,2)`) |
| Classifier | Logistic Regression (`C=1.0`, `solver=lbfgs`) |
| Classes | `0` = Negative, `1` = Positive |
| Expected F1 | ~0.92+ |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Server port (set automatically by Railway) |
| `MODEL_PATH` | `models/sentiment_model.pkl` | Path to trained model |
| `MLFLOW_TRACKING_URI` | `mlruns` | MLflow tracking URI |
