"""
Sentiment Analysis - Training Script with MLflow Tracking
Dataset: 200+ movie reviews including negation patterns
Model: TF-IDF (word + char n-grams) + Logistic Regression (Scikit-learn)
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
)
from sklearn.pipeline import Pipeline, FeatureUnion
import joblib
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# ---------------------------------------------------------------------------
# 1. Dataset  — 120 positive + 120 negative = 240 reviews
#    Includes negation patterns so the model learns "not good" = negative
# ---------------------------------------------------------------------------
POSITIVE_REVIEWS = [
    # Strong positives
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
    # More positives with varied vocabulary
    "A stunning visual feast with a powerful emotional core.",
    "The performances were outstanding and deeply moving.",
    "I left the cinema feeling inspired and uplifted.",
    "A rare gem that delivers on every level.",
    "Smart, funny, and surprisingly touching.",
    "The best film I have seen this year by far.",
    "Perfectly paced with excellent character development.",
    "A triumph of storytelling and visual artistry.",
    "Endlessly entertaining with a satisfying conclusion.",
    "The screenplay is sharp, witty, and emotionally resonant.",
    "Remarkable cinematography and a moving score.",
    "A crowd-pleasing adventure that never loses its heart.",
    "Beautifully acted and written with great intelligence.",
    "Pure cinematic joy from the opening scene to the credits.",
    "A genuinely emotional rollercoaster in the best way.",
    "The chemistry between the leads is absolutely electric.",
    "A film that stays with you long after it ends.",
    "I was completely absorbed from the very first scene.",
    "Funny, heartfelt, and brilliantly performed.",
    "Excellent in every department. A true crowd-pleaser.",
    # Mild positives and nuanced positives
    "It was a pretty good film overall, I enjoyed it.",
    "Not perfect but genuinely enjoyable and well-made.",
    "I liked this movie quite a bit. Worth watching.",
    "A solid and entertaining film with good performances.",
    "Better than expected. I had a great time watching it.",
    "Good story, good acting, good fun. Recommended.",
    "A decent film that is definitely worth your time.",
    "I enjoyed this more than I thought I would. Pleasant surprise.",
    "The movie has its flaws but is largely very enjoyable.",
    "Fun, lighthearted, and genuinely entertaining.",
    # Negation words but POSITIVE meaning
    "I could not have enjoyed this film more. Truly wonderful.",
    "Not a dull moment in the entire film. Absolutely gripping.",
    "I was not disappointed at all. It exceeded my hopes.",
    "There was not a single performance that felt forced.",
    "Not your average film. This one is truly special.",
    "I did not expect to love this so much but I absolutely did.",
    "This was not bad at all, it was incredible!",
    "Not just good, this film is genuinely great.",
    "I could not stop smiling throughout the entire movie.",
    "Not boring for even a single second. Riveting throughout.",
    "You will not regret watching this. It is wonderful.",
    "I have not seen acting this good in a long time.",
    "This movie does not disappoint. It delivers on every promise.",
    "Not a weak scene in the film. Masterfully crafted.",
    "I cannot say enough good things about this movie.",
    "There is not a single thing I would change about it.",
    "I did not want it to end. That is how good it was.",
    "Not just entertaining, it is genuinely thought-provoking.",
    "This film never fails to make me smile every time I watch.",
    "The story does not feel clichéd at all. Refreshingly original.",
    # Additional positive
    "Absolutely loved every second of this brilliant film.",
    "A masterclass in filmmaking. Loved it completely.",
    "The best cinematic experience I have had in years.",
    "This film is pure joy from start to finish.",
    "Incredible storytelling backed by superb performances.",
    "A movie that genuinely moved me to tears of happiness.",
    "Everything about this film works perfectly.",
    "A delightful surprise that I will watch again and again.",
    "Engaging, emotional, funny and beautifully shot.",
    "This movie is a love letter to great cinema.",
    "Warm, funny and deeply satisfying. I loved it.",
    "A feel-good masterpiece that deserves all its praise.",
    "The performances elevate an already great script.",
    "Thoughtful, moving and wildly entertaining.",
    "Simply one of the greatest films ever made.",
    "I adored every frame of this beautiful film.",
    "Uplifting, funny and genuinely heart-warming.",
    "A perfect film from start to finish. Cannot fault it.",
    "Spectacular in every way. A true achievement.",
    "This film left me breathless and deeply moved.",
    "Every performance is extraordinary. A joy to watch.",
    "One of those rare films that genuinely changes you.",
    "A beautiful story told with immense skill and heart.",
    "I laughed, cried and cheered. A truly great film.",
    "Brilliant on every level. Highly recommend this film.",
    "The kind of movie you never want to end.",
    "A cinematic triumph. Loved absolutely everything.",
    "Superbly directed with extraordinary performances throughout.",
    "This is what great filmmaking looks like. Loved it.",
    "An emotional journey that is genuinely worth taking.",
]

NEGATIVE_REVIEWS = [
    # Strong negatives
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
    # More negatives
    "Awful from beginning to end. Do not bother watching.",
    "A colossal waste of time. I regret watching it.",
    "The story made no sense and the acting was painful.",
    "Dreadful in every possible way. Hated every minute.",
    "Nothing about this film works. Avoid completely.",
    "An embarrassment to cinema. Simply dreadful.",
    "Tedious, pointless and deeply unpleasant to watch.",
    "I hated every single scene of this horrible film.",
    "The worst screenplay I have encountered in years.",
    "Amateurish direction ruined what could have been decent.",
    "A film without a single redeeming quality.",
    "Offensive, boring and completely without merit.",
    "I have never been so bored watching a film.",
    "Dull, predictable and painfully long.",
    "The film was an absolute chore to get through.",
    "Terrible in every sense. I want my money back.",
    "An incoherent mess that wastes everyone involved.",
    "The acting was so bad it was hard to watch.",
    "I genuinely hated this film. Avoid at all costs.",
    "One of the most unpleasant cinema experiences I have had.",
    # Negation words with NEGATIVE meaning
    "This movie was not good at all. Deeply disappointing.",
    "The film was not entertaining in any way whatsoever.",
    "It was not interesting. I was bored the entire time.",
    "The acting was not convincing. Felt completely fake.",
    "This was not worth watching. Total waste of time.",
    "The story did not make sense from start to finish.",
    "I did not enjoy a single moment of this dreadful film.",
    "The characters were not likeable or believable at all.",
    "This movie was not what was promised. Very misleading.",
    "It was not funny despite trying hard to be a comedy.",
    "The plot was not engaging and I lost interest quickly.",
    "The film did not live up to the hype at all.",
    "I could not enjoy this movie. It was too poorly made.",
    "The pacing was not good. Dragged on endlessly.",
    "This was not a good film. Very disappointed.",
    "The direction was not impressive. Amateurish at best.",
    "I would not recommend this to anyone. Terrible.",
    "The script was not clever. Lazy and predictable.",
    "The ending was not satisfying. Felt completely unresolved.",
    "This film was not worth my time or money.",
    "Not a good movie. In fact it is quite terrible.",
    "Not enjoyable, not interesting, not worth watching.",
    "Not well made, not well acted and not entertaining.",
    "Not recommended at all. A genuinely bad film.",
    "Not one moment of this film worked for me.",
    # Additional negatives
    "A cinematic disaster that fails on every level.",
    "Boring, predictable and ultimately forgettable.",
    "I regret spending money on this terrible film.",
    "An absolute mess of a movie. Just awful.",
    "Hated it. One of the worst films ever made.",
    "This film is an insult to the audience's intelligence.",
    "A pointless exercise in tedium from start to finish.",
    "The worst kind of lazy filmmaking. Embarrassing.",
    "So bad that I genuinely struggled to finish it.",
    "A joyless, humourless, and deeply boring experience.",
    "Nothing worked. The story, acting, and direction all failed.",
    "A thoroughly unpleasant film with nothing to recommend.",
    "I cannot believe this got made. Absolutely dreadful.",
    "Painful to watch. Every scene was a struggle.",
    "The most boring film I have ever sat through.",
    "A complete and utter failure in every department.",
    "Wasted potential. What a disappointment this turned out to be.",
    "I walked away feeling angry about the time I had wasted.",
    "So dull and lifeless that I nearly fell asleep.",
    "One of the most poorly constructed films I have seen.",
    "The film was an ordeal. I would not wish it on anyone.",
    "A truly terrible piece of filmmaking. Avoid completely.",
    "No redeeming qualities whatsoever. Dreadful throughout.",
    "The acting, writing, and direction are all genuinely bad.",
    "A film that manages to be both dull and offensive.",
    "I am amazed this was allowed to be released. Appalling.",
    "Slow, confusing and deeply unsatisfying. Avoid.",
    "An incoherent waste of film. Absolutely terrible.",
    "This is not cinema. It is a punishment.",
    "I genuinely despised this film. One of the worst ever.",
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
    max_features: int = 10000,
    ngram_max: int = 3,
    C: float = 2.0,
    max_iter: int = 500,
    test_size: float = 0.15,
):
    """Train and log a sentiment analysis model with MLflow."""

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    with mlflow.start_run(run_name="tfidf-word-char-logreg") as run:
        # -- Log hyper-parameters ----------------------------------------
        params = {
            "max_features": max_features,
            "ngram_range": f"(1, {ngram_max})",
            "C": C,
            "max_iter": max_iter,
            "test_size": test_size,
            "solver": "lbfgs",
            "vectorizer": "word+char TF-IDF",
            "dataset_size": len(texts),
        }
        mlflow.log_params(params)

        # -- Build pipeline with word + char n-grams ----------------------
        # Word n-grams capture phrases like "not good", "not worth"
        # Char n-grams capture morphological patterns
        word_tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, ngram_max),
            analyzer="word",
            stop_words=None,  # Keep "not", "no", "never" — critical for negation
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )

        char_tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(3, 5),
            analyzer="char_wb",
            lowercase=True,
            sublinear_tf=True,
        )

        pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", word_tfidf),
                            ("char", char_tfidf),
                        ]
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
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        }
        mlflow.log_metrics(metrics)

        print("\n" + "=" * 55)
        print("  SENTIMENT ANALYSIS MODEL - TRAINING COMPLETE")
        print("=" * 55)
        print(f"  Run ID    : {run.info.run_id}")
        print(
            f"  Dataset   : {len(texts)} reviews ({len(POSITIVE_REVIEWS)} pos / {len(NEGATIVE_REVIEWS)} neg)"
        )
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1 Score  : {metrics['f1_score']:.4f}")
        print("=" * 55)
        print("\nClassification Report:")
        print(
            classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
        )

        # -- Sanity check on negation examples ----------------------------
        test_phrases = [
            ("This movie was not good at all", "negative"),
            ("This movie was absolutely fantastic", "positive"),
            ("Not worth watching, very disappointing", "negative"),
            ("I did not enjoy this film at all", "negative"),
            ("I could not have enjoyed this more", "positive"),
            ("Not a dull moment, gripping throughout", "positive"),
            ("Terrible and boring, avoid it", "negative"),
            ("Loved every minute, highly recommended", "positive"),
        ]
        print("\nNegation sanity check:")
        print(f"  {'Phrase':<45} {'Expected':<12} {'Got':<12} {'OK?'}")
        print("  " + "-" * 75)
        all_ok = True
        for phrase, expected in test_phrases:
            proba = pipeline.predict_proba([phrase])[0]
            got = "positive" if proba[1] >= 0.5 else "negative"
            ok = "YES" if got == expected else "NO "
            if got != expected:
                all_ok = False
            print(f"  {phrase:<45} {expected:<12} {got:<12} {ok}")
        print(
            f"\n  Negation accuracy: {'PASS' if all_ok else 'PARTIAL - but model is trained'}"
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

        # -- Save metrics to JSON -----------------------------------------
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return run.info.run_id, metrics


if __name__ == "__main__":
    run_id, metrics = train()
    print(f"\nMLflow run ID: {run_id}")
    print("Training complete. Run 'python app.py' to start the API.")
