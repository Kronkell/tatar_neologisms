"""Unit tests. Run:  python -m pytest code/tests   (or  python code/tests/test_models.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from data import make_lexicon, sample_comparisons  # noqa: E402
from metrics import equivalent_budget, within_group_tau  # noqa: E402
from models import bt_nll_grad, fit_scores  # noqa: E402


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(10)
    w = rng.integers(0, 10, 50)
    l = (w + rng.integers(1, 10, 50)) % 10
    _, g = bt_nll_grad(s, w, l)
    eps = 1e-6
    num = np.array([(bt_nll_grad(s + eps * e, w, l)[0] - bt_nll_grad(s - eps * e, w, l)[0]) / (2 * eps)
                    for e in np.eye(10)])
    assert np.allclose(g, num, atol=1e-5)


def test_shapes_and_pairs_inside_groups():
    rng = np.random.default_rng(1)
    lex = make_lexicon(rng, 6, 4, 3, rho=0.5, scale=2.0)
    assert lex.X.shape == (24, 3) and lex.u.shape == (24,)
    w, l = sample_comparisons(rng, lex, budget=4)
    assert len(w) == 48
    assert np.all(lex.group[w] == lex.group[l]) and np.all(w != l)


def test_all_methods_recover_ranking_with_many_comparisons():
    rng = np.random.default_rng(2)
    lex = make_lexicon(rng, 10, 6, 5, rho=0.99, scale=2.0)
    w, l = sample_comparisons(rng, lex, budget=200)
    for method in cfg.METHODS:
        s, _ = fit_scores(method, lex.X, w, l, cfg, np.random.default_rng(3))
        assert within_group_tau(lex.u, s, lex.group) > 0.85, method


def test_equivalent_budget_interpolation():
    budgets = [1, 2, 4]
    assert equivalent_budget(budgets, [0.2, 0.4, 0.6], 0.4) == (2.0, False)
    m, cens = equivalent_budget(budgets, [0.2, 0.4, 0.6], 0.5)
    assert np.isclose(m, 2 * np.sqrt(2)) and not cens
    assert equivalent_budget(budgets, [0.2, 0.4, 0.6], 0.9)[1] is True


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
