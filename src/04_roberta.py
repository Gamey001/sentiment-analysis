import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from tqdm import tqdm
from transformers import pipeline

from sentiment_utils import FIGURES, LABELS, MODEL, ROBERTA_LABELS, SCORES_CSV, load_splits

BATCH = 32

df = load_splits()
clf = pipeline("sentiment-analysis", model=MODEL, truncation=True, max_length=512, top_k=None)

texts = df["Text"].tolist()
rows = []
for i in tqdm(range(0, len(texts), BATCH)):
    for out in clf(texts[i:i + BATCH]):
        rows.append({f"p_{ROBERTA_LABELS[o['label']]}": o["score"] for o in out})

probs = pd.DataFrame(rows, index=df.index)
df = pd.concat([df, probs], axis=1)
df["roberta_pred"] = probs[[f"p_{l}" for l in LABELS]].idxmax(axis=1).str.removeprefix("p_")

sia = SentimentIntensityAnalyzer()
vader = pd.DataFrame([sia.polarity_scores(t) for t in df["Text"]], index=df.index).add_prefix("vader_")
df = pd.concat([df, vader], axis=1)

df.drop(columns=["Text", "Summary"]).to_csv(SCORES_CSV, index=False)
print("wrote", SCORES_CSV)

val = df[df["split"] == "val"]
print("RoBERTa, zero-shot (val)")
print(classification_report(val["label"], val["roberta_pred"], digits=3))

FIGURES.mkdir(exist_ok=True)
ConfusionMatrixDisplay.from_predictions(val["label"], val["roberta_pred"], labels=LABELS)
plt.tight_layout()
plt.savefig(FIGURES / "roberta_confusion_val.png", dpi=120)
plt.close()

sns.pairplot(data=val.sample(min(len(val), 1000), random_state=0),
             vars=["vader_neg", "vader_neu", "vader_pos", "p_negative", "p_neutral", "p_positive"],
             hue="Score", palette="tab10")
plt.savefig(FIGURES / "vader_vs_roberta_pairplot.png", dpi=100)
plt.close()

print("\nMost positive-sounding 1-star review (RoBERTa):")
print(val.query("Score == 1").sort_values("p_positive", ascending=False)["Text"].iloc[0])
print("\nMost negative-sounding 5-star review (RoBERTa):")
print(val.query("Score == 5").sort_values("p_negative", ascending=False)["Text"].iloc[0])
