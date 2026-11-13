"""Rankers fitted from pairwise comparisons (winner w beat loser l).

All three are MAP estimates of the Bradley-Terry likelihood
    P(w beats l) = sigmoid(s_w - s_l)
with different parametrisations of the scores s:
    bt            s = eta,                 eta ~ N(0, 1/lambda)       (no features)
    features      s = X theta                                          (features only)
    hierarchical  s = X theta + eta,       eta ~ N(0, 1/lambda)       (proposed)
Added after the first report (assumptions of the theory, lambda fixed, no hold-out):
    bt_fixed      s = eta,                 lambda = 1/V
    hier_oracle   s = mu + eta,            mu = known prior mean, lambda = 1/((1-rho) V)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def bt_nll_grad(s, w, l):
    """Negative log-likelihood of BT and its gradient w.r.t. scores s."""
    z = s[w] - s[l]
    nll = np.sum(np.logaddexp(0.0, -z))
    r = expit(-z)                       # d nll / d z = -r
    n = len(s)
    grad = np.bincount(l, weights=r, minlength=n) - np.bincount(w, weights=r, minlength=n)
    return nll, grad


def _fit_bt(n, w, l, lam):
    def obj(eta):
        nll, g = bt_nll_grad(eta, w, l)
        return nll + 0.5 * lam * eta @ eta, g + lam * eta
    return minimize(obj, np.zeros(n), jac=True, method="L-BFGS-B").x


def _fit_features(X, w, l, lam_theta):
    def obj(theta):
        nll, g = bt_nll_grad(X @ theta, w, l)
        return nll + 0.5 * lam_theta * theta @ theta, X.T @ g + lam_theta * theta
    theta = minimize(obj, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    return X @ theta


def _fit_hierarchical(X, w, l, lam, lam_theta):
    n, d = X.shape

    def obj(p):
        theta, eta = p[:d], p[d:]
        nll, g = bt_nll_grad(X @ theta + eta, w, l)
        val = nll + 0.5 * lam_theta * theta @ theta + 0.5 * lam * eta @ eta
        return val, np.concatenate([X.T @ g + lam_theta * theta, g + lam * eta])

    p = minimize(obj, np.zeros(d + n), jac=True, method="L-BFGS-B").x
    return X @ p[:d] + p[d:]


def _fit_oracle(mu, w, l, lam):
    def obj(eta):
        nll, g = bt_nll_grad(mu + eta, w, l)
        return nll + 0.5 * lam * eta @ eta, g + lam * eta
    return mu + minimize(obj, np.zeros(len(mu)), jac=True, method="L-BFGS-B").x


def mean_nll(s, w, l):
    """Average BT log-loss of scores s on comparisons (w beat l)."""
    if len(w) == 0:
        return np.nan
    return float(np.mean(np.logaddexp(0.0, -(s[w] - s[l]))))


def fit_fixed(method, X, w, l, lam, lam_theta):
    """Fit with a fixed ridge lambda (no hold-out selection)."""
    if method == "bt" or (X.shape[1] == 0 and method == "hierarchical"):
        return _fit_bt(X.shape[0], w, l, lam)
    if method == "hierarchical":
        return _fit_hierarchical(X, w, l, lam, lam_theta)
    if method == "features":
        if X.shape[1] == 0:
            return np.zeros(X.shape[0])
        return _fit_features(X, w, l, lam_theta)
    raise ValueError(method)


def _holdout_loss(s, w, l):
    return bt_nll_grad(s, w, l)[0] / max(len(w), 1)


def select_lambda(fit, w, l, grid, holdout_fraction, rng):
    """Pick the ridge strength with the best held-out BT log-likelihood."""
    n_cmp = len(w)
    n_val = int(round(holdout_fraction * n_cmp))
    if n_val < 2 or n_cmp - n_val < 2:
        return grid[len(grid) // 2]
    perm = rng.permutation(n_cmp)
    val, tr = perm[:n_val], perm[n_val:]
    losses = [_holdout_loss(fit(w[tr], l[tr], lam), w[val], l[val]) for lam in grid]
    return grid[int(np.argmin(losses))]


def fit_scores(method, X, w, l, cfg, rng, prior_mean=None, rho=None):
    """Return (scores, chosen_lambda) for one method.
    bt_fixed / hier_oracle use V = UTILITY_SCALE**2; hier_oracle also needs prior_mean and rho."""
    n = X.shape[0]
    V = cfg.UTILITY_SCALE ** 2
    if method == "bt_fixed":
        return _fit_bt(n, w, l, 1.0 / V), 1.0 / V
    if method == "hier_oracle":
        lam = 1.0 / ((1.0 - rho) * V)
        return _fit_oracle(prior_mean, w, l, lam), lam
    if method == "features" and X.shape[1] == 0:
        return np.zeros(n), np.nan
    if method == "hierarchical" and X.shape[1] == 0:
        method = "bt"
    if method == "features":
        return _fit_features(X, w, l, cfg.LAMBDA_THETA), np.nan
    if method == "bt":
        fit = lambda ww, ll, lam: _fit_bt(n, ww, ll, lam)
    elif method == "hierarchical":
        fit = lambda ww, ll, lam: _fit_hierarchical(X, ww, ll, lam, cfg.LAMBDA_THETA)
    else:
        raise ValueError(method)
    lam = select_lambda(fit, w, l, cfg.LAMBDA_GRID, cfg.HOLDOUT_FRACTION, rng)
    return fit(w, l, lam), lam


def fit_hierarchical_parts(X, w, l, lam, lam_theta):
    """Hierarchical MAP returning (theta, eta) separately (for rho-hat on real data)."""
    n, d = X.shape

    def obj(p):
        theta, eta = p[:d], p[d:]
        nll, g = bt_nll_grad(X @ theta + eta, w, l)
        val = nll + 0.5 * lam_theta * theta @ theta + 0.5 * lam * eta @ eta
        return val, np.concatenate([X.T @ g + lam_theta * theta, g + lam * eta])

    p = minimize(obj, np.zeros(d + n), jac=True, method="L-BFGS-B").x
    return p[:d], p[d:]
