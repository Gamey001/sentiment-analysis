from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RAW_CSV = DATA / "Reviews.csv"
LABELED_CSV = DATA / "reviews_labeled.csv"
SPLITS_CSV = DATA / "splits.csv"
SCORES_CSV = DATA / "roberta_scores.csv"
HAND_LABELS_CSV = DATA / "hand_labels.csv"
METRICS_JSON = ROOT / "results" / "metrics.json"

SEED = 42
SAMPLE_SIZE = 20_000
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ["negative", "neutral", "positive"]
ROBERTA_LABELS = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


def to_label(score):
    if not isinstance(score, (int, np.integer)) or isinstance(score, bool) or not 1 <= score <= 5:
        raise ValueError(f"star score must be an int in 1..5, got {score!r}")
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def vader_label(text, sia, lo=-0.05, hi=0.05):
    c = sia.polarity_scores(text)["compound"]
    return "negative" if c < lo else ("positive" if c > hi else "neutral")


def threshold_predict(p_negative, t):
    return np.where(np.asarray(p_negative) > t, "negative", "not_negative")


def binarize(labels):
    return np.where(np.asarray(labels) == "negative", "negative", "not_negative")


def bootstrap_recall_ci(y_true, y_pred, n_boot=2000, seed=SEED, positive="negative"):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    recalls = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        t, p = y_true[idx], y_pred[idx]
        mask = t == positive
        if mask.any():
            recalls.append((p[mask] == positive).mean())
    return np.percentile(recalls, [2.5, 97.5])


def load_labeled():
    if not LABELED_CSV.exists():
        raise SystemExit(f"{LABELED_CSV} missing — run src/01_load_and_eda.py first")
    return pd.read_csv(LABELED_CSV)


def load_splits():
    if not SPLITS_CSV.exists():
        raise SystemExit(f"{SPLITS_CSV} missing — run src/03_baselines.py first")
    return load_labeled().merge(pd.read_csv(SPLITS_CSV), on="Id")
