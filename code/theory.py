"""Лапласово приближение: предсказание экономии сравнений без подгонки параметров.

Σ(m) = (m·L̄ + a·I)⁻¹,  a_BT = 1/V,  a_H = 1/((1−ρ)V),
L̄ = (N/2) Σ_e q_e Ī_e (e_i − e_j)(e_i − e_j)ᵀ,
R(m, a) = Σ_k w_k / (m·μ_k + a).
"""
from __future__ import annotations

import math

import numpy as np

W_TOL = 1e-12
MU_TOL = 1e-12


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def mean_pq(V: float, n_nodes: int = 200) -> float:
    """Ī = E[σ(D)(1−σ(D))], D ~ N(0, 2V); квадратура Гаусса–Эрмита."""
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    d = math.sqrt(2.0 * 2.0 * V) * x           # D = √(2·Var D)·x для веса e^{−x²}
    p = sigmoid(d)
    return float(np.sum(w * p * (1.0 - p)) / math.sqrt(math.pi))


def saving_complete(rho: float, m0: float, K: int, V: float) -> float:
    """Замкнутая форма для плана random: m_BT = m₀ + (K−1)/(K·Ī·V)·ρ/(1−ρ); вернуть m_BT/m₀."""
    Ibar = mean_pq(V)
    m_bt = m0 + (K - 1) / (K * Ibar * V) * rho / (1.0 - rho)
    return m_bt / m0


def design_pairs(design: str, groups) -> tuple[np.ndarray, np.ndarray]:
    """Пары плана опроса и их вероятности q_e (группа выбирается равновероятно)."""
    pairs, q = [], []
    G = len(groups)
    for S in groups:
        S = np.asarray(S)
        K = len(S)
        if design == "random":
            e = [(S[a], S[b]) for a in range(K) for b in range(a + 1, K)]
        elif design == "cycle":
            e = [(S[a], S[(a + 1) % K]) for a in range(K)] if K > 2 else [(S[0], S[1])]
        elif design == "star":
            e = [(S[0], S[b]) for b in range(1, K)]
        else:
            raise ValueError(design)
        pairs.extend(e)
        q.extend([1.0 / (G * len(e))] * len(e))
    return np.asarray(pairs, dtype=int), np.asarray(q, dtype=float)


def laplacian_from_pairs(pairs, q, Ibar, n: int) -> np.ndarray:
    """L̄ = (N/2) Σ_e q_e Ī_e (e_i − e_j)(e_i − e_j)ᵀ по явному списку пар."""
    pairs = np.asarray(pairs, dtype=int)
    q = np.asarray(q, dtype=float)
    Ie = np.broadcast_to(np.asarray(Ibar, dtype=float), q.shape)
    wgt = (n / 2.0) * q * Ie
    L = np.zeros((n, n))
    i, j = pairs[:, 0], pairs[:, 1]
    np.add.at(L, (i, j), -wgt)
    np.add.at(L, (j, i), -wgt)
    np.add.at(L, (i, i), wgt)
    np.add.at(L, (j, j), wgt)
    return L


def design_laplacian(design, groups, Ibar, n: int | None = None) -> np.ndarray:
    """L̄ для плана опроса.

    design — имя плана ('random', 'cycle', 'star') или кортеж (pairs, q) с эмпирическими
    частотами пар; Ibar — скаляр или массив Ī_e той же длины, что pairs.
    """
    if n is None:
        n = int(max(int(np.max(S)) for S in groups)) + 1
    if isinstance(design, str):
        pairs, q = design_pairs(design, groups)
    else:
        pairs, q = design
    return laplacian_from_pairs(pairs, q, Ibar, n)


def _spectrum(Lbar, groups):
    mu, U = np.linalg.eigh(Lbar)
    mu = np.where(np.abs(mu) < MU_TOL, 0.0, mu)
    w = np.zeros(len(mu))
    for S in groups:
        S = np.asarray(S)
        US = U[S, :]
        w += np.sum(US ** 2, axis=0) - np.sum(US, axis=0) ** 2 / len(S)
    w = np.where(w < W_TOL, 0.0, w)
    return mu, w


def spectrum(Lbar, groups):
    """(μ_k, w_k) для L̄ = U diag(μ) Uᵀ; кэширует разложение по id матрицы."""
    key = id(Lbar)
    if _CACHE.get("key") == key and _CACHE.get("ref") is Lbar:
        return _CACHE["val"]
    val = _spectrum(Lbar, groups)
    _CACHE.update(key=key, ref=Lbar, val=val)
    return val


_CACHE: dict = {}


def risk(m, a, mu, w) -> float:
    """R(m, a) = Σ_k w_k / (m μ_k + a)."""
    return float(np.sum(w / (m * mu + a)))


def equivalent_budget_theory(Lbar, groups, rho, V, m0, m_max=1e4, tol=1e-12):
    """m_BT: R(m_BT, a_BT) = R(m₀, a_H); бисекция по log m на [m₀, m_max]; иначе inf."""
    mu, w = spectrum(Lbar, groups)
    a_bt = 1.0 / V
    a_h = 1.0 / ((1.0 - rho) * V)
    target = risk(m0, a_h, mu, w)
    if risk(m_max, a_bt, mu, w) > target:
        return math.inf
    lo, hi = math.log(m0), math.log(m_max)
    if risk(m0, a_bt, mu, w) <= target:
        return float(m0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if risk(math.exp(mid), a_bt, mu, w) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return math.exp(0.5 * (lo + hi))


def spectral_bounds(Lbar, groups, rho, V, m0):
    """(m₀ + min Δ_k, m₀ + max Δ_k), Δ_k = ρ/((1−ρ)V μ_k) по направлениям с w_k > 0."""
    mu, w = spectrum(Lbar, groups)
    rel = w > W_TOL
    pos = rel & (mu > MU_TOL)
    delta = rho / ((1.0 - rho) * V * mu[pos])
    lo = m0 + float(np.min(delta)) if pos.any() else math.inf
    hi = math.inf if np.any(rel & ~pos) else m0 + float(np.max(delta))
    return lo, hi


# ----------------------------------------------------------------------------- MC
def mc_grid(lo=0.25, hi=512.0, n=90):
    """MC_GRID: n points, log-spaced from lo to hi."""
    return np.exp(np.linspace(math.log(lo), math.log(hi), n))


def _n_cmp(m, n):
    return max(1, int(round(m * n / 2)))


def expected_risk_mc(design, G, K, V, a, m_grid, n_draws=200, mode="full", seed=0):
    """Expected Laplace risk E[R(m, a)] / N under the generative model (no experiment data).

    mode = 'mean_field': analytic L̄ with the average Ī (same as functions 1–4);
           'design':     realised random design (pairs drawn as in sample_comparisons),
                         every comparison carries information Ī;
           'full':       realised design and u ~ N(0, V) i.i.d.; information p_e(1 − p_e).
    In each realisation the design for smaller m is a prefix of the sample at max(m_grid),
    so the risk curve is monotone. L is block-diagonal: K×K blocks are inverted;
    risk = Σ_g tr(C Σ_g) / N, C = I − 11ᵀ/K. `a` may be a scalar or an array of prior precisions;
    the result has shape (len(a), len(m_grid)) (or (len(m_grid),) for scalar a).
    Generator: default_rng([777, seed]).
    """
    from data import sample_pairs  # local import: theory.py stays usable on its own

    scalar = np.ndim(a) == 0
    a = np.atleast_1d(np.asarray(a, dtype=float))
    m_grid = np.asarray(m_grid, dtype=float)
    N = G * K
    Ibar = mean_pq(V)
    if mode == "mean_field":
        groups = [np.arange(g * K, (g + 1) * K) for g in range(G)]
        mu, w = spectrum(design_laplacian(design, groups, Ibar), groups)
        out = np.array([[risk(m, aa, mu, w) / N for m in m_grid] for aa in a])
        return out[0] if scalar else out

    rng = np.random.default_rng([777, seed])
    counts = np.array([_n_cmp(m, N) for m in m_grid])
    n_max = int(counts.max())
    order = np.argsort(counts, kind="stable")
    eye = np.eye(K)
    acc = np.zeros((len(a), len(m_grid)))
    for _ in range(n_draws):
        u = rng.normal(0.0, math.sqrt(V), N) if mode == "full" else None
        g, ia, ib = sample_pairs(rng, G, K, n_max, design)
        if mode == "full":
            p = sigmoid(u[g * K + ia] - u[g * K + ib])
            info = p * (1.0 - p)
        elif mode == "design":
            info = np.full(n_max, Ibar)
        else:
            raise ValueError(mode)
        L = np.zeros((G, K, K))
        done = 0
        for t in order:
            c = counts[t]
            if c > done:
                sl = slice(done, c)
                gg, aa_, bb, ww = g[sl], ia[sl], ib[sl], info[sl]
                np.add.at(L, (gg, aa_, aa_), ww)
                np.add.at(L, (gg, bb, bb), ww)
                np.add.at(L, (gg, aa_, bb), -ww)
                np.add.at(L, (gg, bb, aa_), -ww)
                done = c
            Sig = np.linalg.inv(L[None, :, :, :] + a[:, None, None, None] * eye)
            tr = np.trace(Sig, axis1=2, axis2=3) - Sig.sum(axis=(2, 3)) / K
            acc[:, t] += tr.sum(axis=1) / N
    out = acc / n_draws
    return out[0] if scalar else out


def equivalent_budget_curves(m_grid, risk_bt, risk_h, m0):
    """m_BT / m₀ from risk curves on a grid: E R_BT(m_BT) = E R_H(m₀),
    linear interpolation of log R in log m. inf if not reached on the grid."""
    lm = np.log(np.asarray(m_grid, dtype=float))
    lb = np.log(np.minimum.accumulate(np.asarray(risk_bt, dtype=float)))
    lh = np.log(np.asarray(risk_h, dtype=float))
    target = float(np.interp(math.log(m0), lm, lh))
    if lb[-1] > target:
        return math.inf
    k = int(np.argmax(lb <= target))
    if k == 0:
        return float(np.exp(lm[0])) / m0
    t = (lb[k - 1] - target) / (lb[k - 1] - lb[k]) if lb[k - 1] != lb[k] else 0.0
    return float(math.exp(lm[k - 1] + t * (lm[k] - lm[k - 1]))) / m0
