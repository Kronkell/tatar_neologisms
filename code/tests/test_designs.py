import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import design_edges, make_lexicon, sample_comparisons  # noqa: E402


@pytest.mark.parametrize("design", ["random", "cycle", "star"])
def test_pairs_are_design_edges_within_group(design):
    K = 8
    lex = make_lexicon(np.random.default_rng(0), 7, K, 3, 0.5, 2.0)
    w, l = sample_comparisons(np.random.default_rng(1), lex, 100, design=design)
    assert np.all(w // K == l // K)
    edges = design_edges(design, K)
    got = {tuple(sorted((a % K, b % K))) for a, b in zip(w, l)}
    assert got <= edges
    assert got == edges  # при 2800 сравнениях все рёбра встречаются


def test_random_design_unchanged_rng_order():
    lex = make_lexicon(np.random.default_rng(3), 5, 8, 3, 0.5, 2.0)
    a = sample_comparisons(np.random.default_rng(4), lex, 3)
    b = sample_comparisons(np.random.default_rng(4), lex, 3, design="random")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
