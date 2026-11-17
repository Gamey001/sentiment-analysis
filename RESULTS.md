# Results

> Not yet run. Fill each table from the script output named in its heading. Do not round toward the claim.

Sample: 20,000 of 393,579 deduplicated reviews (568,454 raw), seed 42, 70/15/15 stratified split (14,000 / 3,000 / 3,000). Label mix: 78.0% positive, 14.6% negative, 7.4% neutral.

## Label agreement (`02_hand_label_eval.py`)

| n hand-labelled | Cohen's κ (text label vs star label) |
|---|---|
| 300 | — |

## Validation, 3-class macro F1 (`03_baselines.py`, `04_roberta.py`)

| Model | Macro F1 | Negative recall | Negative precision |
|---|---|---|---|
| Majority class | 0.292 | 0.000 | 0.000 |
| VADER (±0.05) | 0.472 | 0.411 | 0.557 |
| RoBERTa zero-shot (argmax) | — | — | — |

## Negative detection, test set (`05_threshold_and_test.py`)

Threshold chosen on val only: **—**

| Precision | Recall | F1 | Recall 95% CI (bootstrap, 2000×) |
|---|---|---|---|
| — | — | — | [—, —] |

## Negative-review topics (`06_topics.py`)

| Topic | Size | Top words | My name for it |
|---|---|---|---|
| — | — | — | — |
