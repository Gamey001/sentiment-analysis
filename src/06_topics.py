import pandas as pd
from bertopic import BERTopic
from umap import UMAP

from sentiment_utils import DATA, SCORES_CSV, SEED, load_labeled

if not SCORES_CSV.exists():
    raise SystemExit(f"{SCORES_CSV} missing — run src/04_roberta.py first")

scores = pd.read_csv(SCORES_CSV)[["Id", "roberta_pred"]]
df = load_labeled().merge(scores, on="Id")
neg = df[df["roberta_pred"] == "negative"]["Text"].tolist()
print(f"{len(neg)} reviews predicted negative")

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
topic_model = BERTopic(min_topic_size=25, umap_model=umap_model, verbose=True)
topics, _ = topic_model.fit_transform(neg)
print(topic_model.get_topic_info().head(15).to_string())

topic_model.get_document_info(neg).to_csv(DATA / "topics.csv", index=False)
print("wrote", DATA / "topics.csv")
