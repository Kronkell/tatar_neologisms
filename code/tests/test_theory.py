import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import theory  # noqa: E402


def _groups(G, K):
    return [np.arange(g * K, (g + 1) * K) for g in range(G)]


def test_mean_pq_v4():
    assert abs(theory.mean_pq(4.0) - 0.120) < 0.0015


def test_mean_pq_limits():
    assert abs(theory.mean_pq(1e-8) - 0.25) < 1e-6
    assert theory.mean_pq(100.0) < theory.mean_pq(4.0)


def test_complete_constant():
    Ibar = theory.mean_pq(4.0)
    c = 7 / (8 * Ibar * 4.0)
    assert abs(c - 1.82) < 0.01


@pytest.mark.parametrize("rho", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("m0", [1, 2, 4])
def test_random_matches_closed_form(rho, m0):
    G, K, V = 10, 8, 4.0
    groups = _groups(G, K)
    L = theory.design_laplacian("random", groups, theory.mean_pq(V))
    m_bt = theory.equivalent_budget_theory(L, groups, rho, V, m0)
    closed = theory.saving_complete(rho, m0, K, V)
    assert abs(m_bt / m0 - closed) / closed < 1e-6
    lo, hi = theory.spectral_bounds(L, groups, rho, V, m0)
    assert lo - 1e-9 <= m_bt <= hi + 1e-9


@pytest.mark.parametrize("design", ["random", "cycle", "star"])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("m0", [1, 4])
def test_solution_inside_bounds(design, rho, m0):
    G, K, V = 6, 8, 4.0
    groups = _groups(G, K)
    L = theory.design_laplacian(design, groups, theory.mean_pq(V))
    m_bt = theory.equivalent_budget_theory(L, groups, rho, V, m0)
    lo, hi = theory.spectral_bounds(L, groups, rho, V, m0)
    assert lo - 1e-9 <= m_bt <= hi + 1e-9


def test_sparse_designs_save_more():
    G, K, V, rho, m0 = 6, 8, 4.0, 0.5, 1
    groups = _groups(G, K)
    Ibar = theory.mean_pq(V)
    s = {d: theory.equivalent_budget_theory(theory.design_laplacian(d, groups, Ibar),
                                            groups, rho, V, m0) for d in ["random", "cycle", "star"]}
    assert s["cycle"] > s["random"] and s["star"] > s["random"]


def test_unobserved_contrast_gives_inf_upper_bound():
    # группа из 3 объектов, сравнивается только пара (0, 1): контраст объекта 2 не наблюдается
    groups = [np.array([0, 1, 2])]
    L = theory.laplacian_from_pairs(np.array([[0, 1]]), np.array([1.0]), 0.12, 3)
    lo, hi = theory.spectral_bounds(L, groups, 0.5, 4.0, 1)
    assert math.isfinite(lo) and hi == math.inf


def test_risk_matches_direct_trace():
    G, K, V, m, a = 3, 5, 4.0, 2.0, 0.3
    groups = _groups(G, K)
    L = theory.design_laplacian("cycle", groups, 0.12)
    Sigma = np.linalg.inv(m * L + a * np.eye(G * K))
    direct = sum(np.trace(Sigma[np.ix_(S, S)]) - Sigma[np.ix_(S, S)].sum() / len(S) for S in groups)
    mu, w = theory.spectrum(L, groups)
    assert abs(theory.risk(m, a, mu, w) - direct) < 1e-9


@pytest.mark.parametrize("design", ["random", "cycle", "star"])
@pytest.mark.parametrize("mode", ["design", "full"])
def test_mc_risk_not_below_mean_field(design, mode):
    G, K, V = 6, 8, 4.0
    grid = theory.mc_grid(0.25, 64, 12)
    a = np.array([1 / V, 1 / (0.5 * V)])
    mf = theory.expected_risk_mc(design, G, K, V, a, grid, mode="mean_field")
    mc = theory.expected_risk_mc(design, G, K, V, a, grid, n_draws=20, mode=mode, seed=1)
    assert np.all(mc >= mf * (1 - 1e-12))


def test_curve_saving_matches_spectral_solution():
    G, K, V, rho = 6, 8, 4.0, 0.5
    groups = _groups(G, K)
    grid = theory.mc_grid()
    a = np.array([1 / V, 1 / ((1 - rho) * V)])
    R = theory.expected_risk_mc("random", G, K, V, a, grid, mode="mean_field")
    L = theory.design_laplacian("random", groups, theory.mean_pq(V))
    for m0 in [1, 2, 4]:
        exact = theory.equivalent_budget_theory(L, groups, rho, V, m0) / m0
        assert abs(theory.equivalent_budget_curves(grid, R[0], R[1], m0) / exact - 1) < 2e-3
