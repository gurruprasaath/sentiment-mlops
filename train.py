"""
Sentiment Analysis - Training Script with MLflow Tracking
Dataset: IMDb-style movie review dataset (generated inline for self-contained demo)
Model: TF-IDF + Logistic Regression (Scikit-learn)
"""

import os
import json
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Sample Dataset (positive & negative movie reviews)
# ---------------------------------------------------------------------------
POSITIVE_REVIEWS = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "An incredible performance by the entire cast. Truly breathtaking.",
    "One of the best films I have ever seen. Highly recommended!",
    "The storyline was gripping and the visuals were stunning.",
    "A masterpiece of modern cinema. Brilliant direction.",
    "I was completely blown away by this film. Amazing!",
    "Wonderful movie, great acting, and an uplifting story.",
    "Loved it! The plot twists kept me on the edge of my seat.",
    "This is a must-watch. Exceptional storytelling.",
    "Superb! Every scene was crafted with care and passion.",
    "Heartwarming and funny. Perfect for the whole family.",
    "The characters were so well written and relatable.",
    "Outstanding! This movie exceeded all my expectations.",
    "A feel-good film that left me smiling for days.",
    "Brilliant script, flawless acting, and gorgeous cinematography.",
    "Truly inspiring. This film touched my heart deeply.",
    "Five stars easily! One of the best of the decade.",
    "Exciting, emotional, and beautifully directed.",
    "An absolute delight from start to finish.",
    "I cannot recommend this movie enough. Pure gold.",
    "Great movie! Loved the chemistry between the leads.",
    "A beautiful and moving experience. Simply wonderful.",
    "The acting was top-notch and the plot was riveting.",
    "This film made me laugh, cry, and cheer. Phenomenal!",
    "I was thoroughly entertained from beginning to end.",
    "A joyful, funny, and deeply touching movie.",
    "Best film of the year without a doubt. Spectacular!",
    "The director's vision comes through in every frame.",
    "Impressive storytelling with incredible character depth.",
    "Loved every moment. A true cinematic gem.",
]

NEGATIVE_REVIEWS = [
    "Terrible movie. I wasted two hours of my life.",
    "The plot made absolutely no sense. Very disappointing.",
    "Horrible acting and a boring, predictable storyline.",
    "I walked out halfway through. That bad.",
    "One of the worst films I have ever seen. Avoid it.",
    "Complete waste of money. The script was dreadful.",
    "Poorly directed, poorly acted, and painfully slow.",
    "I fell asleep. Nothing interesting happens at all.",
    "The characters were flat and the dialogue was cringeworthy.",
    "A total disaster from start to finish. Not recommended.",
    "Dull, lifeless, and utterly forgettable.",
    "The special effects were laughable and the story was weak.",
    "I cannot believe how bad this movie was. Shocking.",
    "A complete mess. The editing was atrocious.",
    "Very disappointing. The trailer was far better than the film.",
    "Boring and tedious. I kept checking the time.",
    "The worst movie I have seen in years. Truly awful.",
    "Terrible pacing and zero emotional impact.",
    "Bad acting all around. I could not connect with any character.",
    "Save yourself the trouble and skip this one entirely.",
    "The movie dragged on forever with no payoff.",
    "Nonsensical plot with terrible CGI. Hard to watch.",
    "A poorly executed idea that had potential but failed.",
    "The dialogue was so bad it was almost funny.",
    "Absolutely dreadful. A new low for the studio.",
    "The worst two hours I have spent in a cinema.",
    "Nothing works in this film. Poor in every way.",
    "A confusing, dull, and ultimately pointless experience.",
    "I was bored to tears. Not even worth streaming for free.",
    "Terrible movie that ruined a great source material.",
]

texts = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
labels = [1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS)


# ---------------------------------------------------------------------------
# 2. MLflow Experiment
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "sentiment-analysis"
MODEL_NAME = "sentiment-classifier"

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment(EXPERIMENT_NAME)


def train(
    max_features: int = 5000,
    ngram_max: int = 2,
    C: float = 1.0,
    max_iter: int = 200,
    test_size: float = 0.2,
):
    """Train and log a sentiment analysis model with MLflow."""

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    with mlflow.start_run(run_name="tfidf-logreg") as run:
        # -- Log hyper-parameters ----------------------------------------
        params = {
            "max_features": max_features,
            "ngram_range": f"(1, {ngram_max})",
            "C": C,
            "max_iter": max_iter,
            "test_size": test_size,
            "solver": "lbfgs",
        }
        mlflow.log_params(params)

        # -- Build pipeline -----------------------------------------------
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=(1, ngram_max),
                        stop_words="english",
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )

        # -- Train --------------------------------------------------------
        pipeline.fit(X_train, y_train)

        # -- Evaluate -----------------------------------------------------
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)

        print("\n" + "=" * 55)
        print("  SENTIMENT ANALYSIS MODEL - TRAINING COMPLETE")
        print("=" * 55)
        print(f"  Run ID : {run.info.run_id}")
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print("=" * 55)
        print("\nClassification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
        )

        # -- Log the model ------------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        # -- Save model locally for the API -------------------------------
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/sentiment_model.pkl")
        print("\nModel saved to models/sentiment_model.pkl")

        # -- Save metrics to JSON (useful for CI checks) ------------------
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return run.info.run_id, metrics


if __name__ == "__main__":
    run_id, metrics = train()
    print(f"\nMLflow run ID: {run_id}")
    print("Training complete. Run 'python app.py' to start the API.")
