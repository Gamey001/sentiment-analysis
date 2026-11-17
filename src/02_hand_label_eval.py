import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from sentiment_utils import HAND_LABELS_CSV, LABELS, SEED, load_labeled

N = 300

if not HAND_LABELS_CSV.exists():
    df = load_labeled()
    per_class = N // len(LABELS)
    sample = df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(per_class, len(g)), random_state=SEED))
    sample = sample.sample(frac=1, random_state=SEED)
    # Star label is withheld so labelling is blind to the rating.
    sample[["Id", "Text"]].assign(my_label="").to_csv(HAND_LABELS_CSV, index=False)
    raise SystemExit(
        f"Wrote {len(sample)} reviews to {HAND_LABELS_CSV}.\n"
        f"Fill my_label with one of {LABELS} from the text alone, then rerun.")

hand = pd.read_csv(HAND_LABELS_CSV).dropna(subset=["my_label"])
hand["my_label"] = hand["my_label"].str.strip().str.lower()
bad = set(hand["my_label"]) - set(LABELS)
if bad:
    raise SystemExit(f"unexpected labels in my_label: {bad}")

merged = hand.merge(load_labeled()[["Id", "label"]], on="Id")
print(f"{len(merged)} hand-labelled reviews")

kappa = cohen_kappa_score(merged["my_label"], merged["label"])
print(f"Cohen's kappa (my text label vs star label): {kappa:.3f}")
print(pd.DataFrame(confusion_matrix(merged["my_label"], merged["label"], labels=LABELS),
                   index=[f"me:{l}" for l in LABELS], columns=[f"stars:{l}" for l in LABELS]))
