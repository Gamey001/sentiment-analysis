# Results

> Steps 01 and 03–06 have been run. Hand-label agreement (02) is pending, and the topic names column is yours to fill in.

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
| RoBERTa zero-shot (argmax) | 0.629 | 0.760 | 0.687 |

RoBERTa barely helps on neutral (F1 0.236 vs VADER 0.055). Three-star reviews are mixed in tone, and the model was trained on tweets.

## Negative detection, test set (`05_threshold_and_test.py`)

Threshold chosen on val only: **p_negative > 0.45** (val F1 0.731, effectively tied with 0.35).

| Precision | Recall | F1 | Recall 95% CI (bootstrap, 2000×) |
|---|---|---|---|
| 0.700 | 0.694 | 0.697 | [0.651, 0.734] |

Test F1 is 0.034 below val, the usual optimism from choosing the threshold on val. Lowering the threshold to 0.20 raises val recall to 0.863 at precision 0.582, which fits better if missing a complaint costs more than reading a false alarm.

Where the model and the stars disagree most, the star rating is often the thing that's wrong. The most positive-scored 1★ review reads "It was awesome. thamk you for prompt delivery *****", which is very likely a mis-click on the rating.

## Negative-review topics (`06_topics.py`)

3,128 reviews RoBERTa predicted negative; 1,074 (34%) were left as outliers (topic -1).

| Topic | Size | Top words | My name for it |
|---|---|---|---|
| 0 | 344 | coffee, cup, kcups, like | |
| 1 | 297 | dog, food, dogs, cat | |
| 2 | 174 | chips, like, taste, eat | |
| 3 | 157 | tea, teas, green, flavor | |
| 4 | 130 | cookies, bread, gluten, cake | |
| 5 | 100 | taste, flavor, product, bitter | |
| 6 | 92 | price, pack, amazon, received | |
| 7 | 89 | bars, chocolate, bar, taste | |
| 8 | 83 | cereal, box, cereals, oats | |
| 9 | 81 | sugar, splenda, syrup, sweeteners | |
| 10 | 54 | item, order, flowers, fedex | |
| 11 | 48 | sauce, hot, stuff, hot sauce | |
| 12 | 45 | candy, candies, melted, chocolate | |
| 13 | 45 | noodles, pasta, sauce, noodle | |
| 14 | 45 | soup, miso, broth, taste | |
| 15 | 44 | mole, flies, house, moth | |
| 16 | 40 | date, expiration, expiration date, expired | |
| 17 | 38 | salt, sodium, natural, salty | |
| 18 | 38 | nuts, almonds, raw, almond | |
| 19 | 31 | gum, flavor, chewing, chew | |
| 20 | 29 | jerky, beef, beef jerky, teriyaki | |
| 21 | 25 | tuna, salmon, fish, cat | |
| 22 | 25 | cans, dented, beans, refried beans | |

Most clusters are grouped by product (coffee, dog food, tea). Only a few are grouped by the kind of complaint: shipping/delivery (6, 10), expired stock (16), pantry moths (15), dented cans (22), melted candy (12).
