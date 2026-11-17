import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd

from sentiment_utils import FIGURES, LABELED_CSV, RAW_CSV, SAMPLE_SIZE, SEED, to_label

if not RAW_CSV.exists():
    raise SystemExit(
        f"{RAW_CSV} missing — run src/00_fetch_data.py first")

df = pd.read_csv(RAW_CSV)
print("raw:", df.shape)

# Same review text is often posted under several product variants; keep one to avoid split leakage.
df = df.drop_duplicates(subset=["UserId", "Time", "Text"]).drop_duplicates(subset="Text")
print("deduplicated:", df.shape)

if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=SEED)
print("sampled:", df.shape)

df["label"] = df["Score"].map(to_label)
print(df["label"].value_counts(normalize=True).round(3))

FIGURES.mkdir(exist_ok=True)
ax = df["Score"].value_counts().sort_index().plot(
    kind="bar", title="Count of Reviews by Stars", figsize=(10, 5))
ax.set_xlabel("Review Stars")
plt.tight_layout()
plt.savefig(FIGURES / "stars.png", dpi=120)
plt.close()

example = "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go."
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
print(tagged[:10])
nltk.chunk.ne_chunk(tagged).pprint()

df[["Id", "Score", "label", "Time", "Summary", "Text"]].to_csv(LABELED_CSV, index=False)
print("wrote", LABELED_CSV)
