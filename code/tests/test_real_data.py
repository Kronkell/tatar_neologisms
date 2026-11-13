import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import real_data  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return real_data.load_real()


def test_control_counts(data):
    assert real_data.control_counts(data) == config.EXPECTED_COUNTS


def test_groups_are_valid(data):
    for S in data.groups:
        assert len(S) >= config.MIN_GROUP_SIZE
        assert len(set(S.tolist())) == len(S)
    assert all(" " not in w and w == w.lower() for w in data.words)


def test_wcl_keeps_literal_null_nan(tmp_path):
    p = tmp_path / "wcl.tsv"
    p.write_text("null\t1\t2\t-1\t1\t1\t1\t1\t1\t1\t1\t1\nNaN\t3\t3\t3\t3\t3\t3\t3\t3\t3\t3\t3\n"
                 "nan\t2\t2\t2\t2\t2\t2\t2\t2\t2\t2\t2\n", encoding="utf-8")
    r, dups = real_data.load_wcl(p)
    assert set(r) == {"null", "nan"} and dups == 1
    assert r["nan"][0] == 2  # последнее вхождение


def test_lambda_selection_holds_out_one_annotator():
    import numpy as np
    import real_exp
    w = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    l = np.array([1, 2, 0, 2, 0, 1, 1, 2])
    who = np.array([3, 3, 5, 5, 3, 7, 7, 3])
    seen = []

    def fit(ww, ll, lam):
        seen.append(len(ww))
        return np.zeros(3) + lam * np.arange(3)

    lam, losses = real_exp.select_lambda_annotator(fit, w, l, who, 3, [0.1, 1.0])
    assert seen == [4, 4]          # trained without the 4 comparisons of annotator 3
    assert lam in (0.1, 1.0) and len(losses) == 2
    assert real_exp.holdout_annotator(0, 2, np.array([1, 4, 6])) in (1, 4, 6)
