import numpy as np
import pytest

from sentiment_utils import threshold_predict, to_label


@pytest.mark.parametrize("score,expected", [
    (1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive"),
    (np.int64(2), "negative"),
])
def test_star_mapping(score, expected):
    assert to_label(score) == expected


@pytest.mark.parametrize("bad", [0, 6, -1, 2.5, "3", None, True])
def test_star_mapping_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        to_label(bad)


def test_threshold_is_strictly_greater_than():
    p = [0.0, 0.29, 0.3, 0.31, 1.0]
    assert threshold_predict(p, 0.3).tolist() == [
        "not_negative", "not_negative", "not_negative", "negative", "negative"]


def test_threshold_extremes():
    p = np.array([0.0, 0.5, 1.0])
    assert (threshold_predict(p, -0.01) == "negative").all()
    assert (threshold_predict(p, 1.0) == "not_negative").all()
