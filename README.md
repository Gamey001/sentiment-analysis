# Review Sentiment
Sentinment Analysis run on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset.

## Problem

Can we flag negative product reviews from their text alone, and what are those negative reviews about?

The original notebook scores 500 reviews with VADER and `cardiffnlp/twitter-roberta-base-sentiment`, then compares the scores by eye. This project keeps both models and adds the parts needed to make a claim about them: a held-out test set, baselines, a tuned decision threshold, confidence intervals, a check on the labels themselves, and topic modelling of the negatives.

## Method

| Step | Script | What it does |
|---|---|---|
| 0 | `src/00_fetch_data.py` | Download the original [SNAP release](https://snap.stanford.edu/data/web-FineFoods.html) and convert it to the Kaggle `Reviews.csv` layout (568,454 rows, same order) |
| 1 | `src/01_load_and_eda.py` | Load, deduplicate, sample 20k reviews (seed 42), map stars to labels (1–2 negative, 3 neutral, 4–5 positive), plus the notebook's NLTK tokenise/POS/NER demo |
| 2 | `src/02_hand_label_eval.py` | Write 300 stratified reviews, with the star rating hidden, for hand labelling. On the second run it computes Cohen's κ between the hand labels and the star labels |
| 3 | `src/03_baselines.py` | Stratified 70/15/15 split. Baselines: majority class and VADER (±0.05 compound band) |
| 4 | `src/04_roberta.py` | Zero-shot RoBERTa class probabilities, validation report, confusion matrix, VADER-vs-RoBERTa pairplot, most-disagreeing examples |
| 5 | `src/05_threshold_and_test.py` | Sweep the `p_negative` threshold on val, freeze the best-F1 threshold, score the test set once, bootstrap a 95% CI on recall |
| 6 | `src/06_topics.py` | BERTopic on the reviews predicted negative, written to `data/topics.csv` |

## Running it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['vader_lexicon','punkt_tab','averaged_perceptron_tagger_eng','maxent_ne_chunker_tab','words']]"

python src/00_fetch_data.py       # public SNAP source, no Kaggle account needed
python src/01_load_and_eda.py
python src/02_hand_label_eval.py   # writes data/hand_labels.csv; fill my_label, rerun
python src/03_baselines.py
python src/04_roberta.py           # roughly 20–40 min on CPU for 20k reviews
python src/05_threshold_and_test.py
python src/06_topics.py

pytest
```

Python 3.12 is used because PyTorch wheels lag new CPython releases. `SAMPLE_SIZE` in `src/sentiment_utils.py` sets how many reviews are used. Set it to `None` for all ~390k deduplicated reviews.

## Results

See [RESULTS.md](RESULTS.md).

## Tests

`tests/test_labels.py` covers the star-to-label mapping (all five stars, numpy ints, and rejection of 0, 6, floats, strings, `None` and `bool`) and the threshold function (the strict `>` boundary and the extremes). `tests/test_vader.py` checks the VADER wrapper on strings with known sentiment, including the notebook's own examples, and checks that the ±0.05 neutral band edges are exclusive.

## Honest limitations

- **Single dataset.** Everything is measured on Amazon Fine Food Reviews. Nothing here shows the results carry over to other products, platforms or writing styles.
- **Ten-year-old data.** The reviews date from 1999–2012. Language, products and review culture have changed since then, and the twitter-RoBERTa model was trained on a different time and domain.
- **Ground truth comes from star ratings, not from the sentiment of the text.** A 1-star review can be written in mild or even positive language, and a 5-star review can grumble. The hand-labelled κ in step 2 measures how far apart those two things are, but the headline metrics are still scored against stars.
- **No business outcome can be measured.** There is no data on what happens after a review is flagged (refunds, churn, response time), so precision and recall cannot be turned into value or cost.
- **The topic names are my judgement.** BERTopic produces clusters of keywords. The human-readable names given to them in RESULTS.md are my interpretation, and someone else could reasonably name or merge them differently.
