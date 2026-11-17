import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sentiment_utils import FIGURES, SEED, SPLITS_CSV, load_labeled, vader_label

plt.style.use("ggplot")
df = load_labeled()

train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=SEED)
pd.concat([
    train[["Id"]].assign(split="train"),
    val[["Id"]].assign(split="val"),
    test[["Id"]].assign(split="test"),
]).to_csv(SPLITS_CSV, index=False)
print(f"train={len(train)} val={len(val)} test={len(test)}")

majority = train["label"].value_counts().idxmax()
print(f"\nBaseline 1: always '{majority}' (val)")
print(classification_report(val["label"], [majority] * len(val), digits=3, zero_division=0))

sia = SentimentIntensityAnalyzer()
val = val.copy()
val["vader_pred"] = val["Text"].map(lambda t: vader_label(t, sia))
print("Baseline 2: VADER, no training (val)")
print(classification_report(val["label"], val["vader_pred"], digits=3))

scores = pd.DataFrame([sia.polarity_scores(t) for t in val["Text"]], index=val.index)
vaders = pd.concat([val[["Score"]], scores], axis=1)

FIGURES.mkdir(exist_ok=True)
ax = sns.barplot(data=vaders, x="Score", y="compound")
ax.set_title("VADER Compound Score by Star Rating")
plt.tight_layout()
plt.savefig(FIGURES / "vader_compound.png", dpi=120)
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
for ax, col, title in zip(axs, ["pos", "neu", "neg"], ["Positive", "Neutral", "Negative"]):
    sns.barplot(data=vaders, x="Score", y=col, ax=ax)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(FIGURES / "vader_pos_neu_neg.png", dpi=120)
plt.close()
