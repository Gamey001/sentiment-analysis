import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from sentiment_utils import METRICS_JSON, SCORES_CSV, binarize, bootstrap_recall_ci, threshold_predict

if not SCORES_CSV.exists():
    raise SystemExit(f"{SCORES_CSV} missing — run src/04_roberta.py first")
df = pd.read_csv(SCORES_CSV)
val, test = df[df["split"] == "val"], df[df["split"] == "test"]


def neg_prf(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        binarize(y_true), y_pred, labels=["negative"], average=None, zero_division=0)
    return p[0], r[0], f[0]


print("threshold sweep on val (negative vs not_negative)")
sweep = []
for t in np.arange(0.2, 0.7, 0.05):
    p, r, f = neg_prf(val["label"], threshold_predict(val["p_negative"], t))
    sweep.append((round(float(t), 2), p, r, f))
    print(f"{t:.2f}  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}")

best_t = max(sweep, key=lambda s: s[3])[0]
print(f"\nchosen threshold (max val F1): {best_t}")

# Test set is touched exactly once, with the threshold frozen from val.
test_pred = threshold_predict(test["p_negative"], best_t)
p, r, f = neg_prf(test["label"], test_pred)
lo, hi = bootstrap_recall_ci(binarize(test["label"]), test_pred)
print(f"test  precision={p:.3f}  recall={r:.3f}  f1={f:.3f}  recall 95% CI=[{lo:.3f}, {hi:.3f}]")

METRICS_JSON.parent.mkdir(exist_ok=True)
METRICS_JSON.write_text(json.dumps({
    "threshold": best_t,
    "val_sweep": [dict(zip(["t", "precision", "recall", "f1"], s)) for s in sweep],
    "test": {"precision": p, "recall": r, "f1": f, "recall_ci95": [lo, hi], "n": len(test)},
}, indent=2, default=float))
print("wrote", METRICS_JSON)
