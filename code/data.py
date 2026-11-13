"""Synthetic lexicon and simulated pairwise judgements of native speakers."""
from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class Lexicon:
    X: np.ndarray        # (N, d) linguistic features of candidates
    u: np.ndarray        # (N,) true latent utility ("how much speakers like it")
    group: np.ndarray    # (N,) index of the loanword the candidate replaces
    n_groups: int
    n_candidates: int
    prior_mean: np.ndarray = None  # scale*sqrt(rho)*f: part of u explained by features (oracle)


def make_lexicon(rng, n_groups, n_candidates, n_features, rho, scale):
    """u_i = scale * (sqrt(rho) * f_i + sqrt(1 - rho) * eta_i),
    f = standardized theta^T x  (part explained by features),
    eta ~ N(0, 1)               (item-specific taste not captured by features)."""
    n = n_groups * n_candidates
    X = rng.standard_normal((n, n_features))
    theta = rng.standard_normal(n_features)
    f = X @ theta
    f = (f - f.mean()) / f.std()
    eta = rng.standard_normal(n)
    u = scale * (np.sqrt(rho) * f + np.sqrt(1.0 - rho) * eta)
    group = np.repeat(np.arange(n_groups), n_candidates)
    return Lexicon(X, u, group, n_groups, n_candidates,
                   prior_mean=scale * np.sqrt(rho) * f)


def design_edges(design, k):
    """Within-group edges of a survey design (local candidate indices, a < b)."""
    if design == "random":
        return {(a, b) for a in range(k) for b in range(a + 1, k)}
    if design == "cycle":
        return {tuple(sorted((a, (a + 1) % k))) for a in range(k)}
    if design == "star":
        return {(0, b) for b in range(1, k)}
    raise ValueError(design)


def sample_pairs(rng, n_groups, k, n_cmp, design="random"):
    """Group index and local candidates (a, b) of n_cmp comparisons of a design.
    random: a random pair; cycle: a random edge (a, a+1 mod K); star: (0, b), b in 1..K-1.
    For "random" the generator calls are exactly those of the original code."""
    g = rng.integers(0, n_groups, n_cmp)
    if design == "random":
        a = rng.integers(0, k, n_cmp)
        b = (a + rng.integers(1, k, n_cmp)) % k
    elif design == "cycle":
        a = rng.integers(0, k, n_cmp)
        b = (a + 1) % k
    elif design == "star":
        a = np.zeros(n_cmp, dtype=np.int64)
        b = rng.integers(1, k, n_cmp)
    else:
        raise ValueError(design)
    return g, a, b


def sample_comparisons(rng, lex, budget, design="random"):
    """Draw round(budget * N / 2) random pairs inside loanword groups
    (each comparison touches two candidates, so `budget` is the average number
    of comparisons per candidate) and Bradley-Terry outcomes
    P(i beats j) = sigmoid(u_i - u_j). Returns (winners, losers)."""
    n = len(lex.u)
    n_cmp = max(1, int(round(budget * n / 2)))
    k = lex.n_candidates
    g, a, b = sample_pairs(rng, lex.n_groups, k, n_cmp, design)
    i, j = g * k + a, g * k + b
    i_wins = rng.random(n_cmp) < expit(lex.u[i] - lex.u[j])
    return np.where(i_wins, i, j), np.where(i_wins, j, i)
