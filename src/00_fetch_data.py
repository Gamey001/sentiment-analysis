import gzip
import urllib.request

import pandas as pd

from sentiment_utils import DATA, RAW_CSV

URL = "https://snap.stanford.edu/data/finefoods.txt.gz"
GZ = DATA / "finefoods.txt.gz"
FIELDS = {
    "product/productId": "ProductId",
    "review/userId": "UserId",
    "review/profileName": "ProfileName",
    "review/helpfulness": "helpfulness",
    "review/score": "Score",
    "review/time": "Time",
    "review/summary": "Summary",
    "review/text": "Text",
}


def parse(path):
    record = {}
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if record:
                    yield record
                    record = {}
                continue
            key, _, value = line.partition(": ")
            if key in FIELDS:
                record[FIELDS[key]] = value
    if record:
        yield record


DATA.mkdir(exist_ok=True)
if not GZ.exists():
    print("downloading", URL)
    urllib.request.urlretrieve(URL, GZ)

df = pd.DataFrame(parse(GZ))
df.insert(0, "Id", range(1, len(df) + 1))
helpful = df.pop("helpfulness").str.split("/", expand=True).astype(int)
df.insert(4, "HelpfulnessNumerator", helpful[0])
df.insert(5, "HelpfulnessDenominator", helpful[1])
df["Score"] = df["Score"].astype(float).astype(int)
df["Time"] = df["Time"].astype(int)

df.to_csv(RAW_CSV, index=False)
print(f"wrote {RAW_CSV}: {df.shape}")
print(df.head(3)[["Id", "ProductId", "ProfileName", "Score", "Summary"]].to_string())
