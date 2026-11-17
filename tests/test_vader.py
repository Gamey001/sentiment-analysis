import pytest
from nltk.sentiment import SentimentIntensityAnalyzer

from sentiment_utils import vader_label


@pytest.fixture(scope="module")
def sia():
    return SentimentIntensityAnalyzer()


@pytest.mark.parametrize("text,expected", [
    ("I am so happy!", "positive"),
    ("Great taffy at a great price.", "positive"),
    ("This is the worst thing ever.", "negative"),
    ("This oatmeal is not good. Its mushy, soft, I don't like it.", "negative"),
    ("The box arrived on Tuesday.", "neutral"),
])
def test_known_strings(sia, text, expected):
    assert vader_label(text, sia) == expected


def test_band_edges_are_exclusive(sia):
    class Fixed:
        def __init__(self, c):
            self.c = c

        def polarity_scores(self, _):
            return {"compound": self.c}

    assert vader_label("", Fixed(0.05)) == "neutral"
    assert vader_label("", Fixed(-0.05)) == "neutral"
    assert vader_label("", Fixed(0.0501)) == "positive"
    assert vader_label("", Fixed(-0.0501)) == "negative"
