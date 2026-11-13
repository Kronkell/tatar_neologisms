"""Quality criteria and the budget-saving analysis."""
import numpy as np
from scipy.stats import kendalltau


def within_group_tau(u_true, scores, group):
    """Mean over loanwords of Kendall tau between true and estimated
    rankings of its candidates (external criterion)."""
    taus = []
    for g in np.unique(group):
        mask = group == g
        t = kendalltau(u_true[mask], scores[mask]).statistic
        taus.append(0.0 if np.isnan(t) else t)   # constant scores = no information
    return float(np.mean(taus))


def equivalent_budget(budgets, tau_curve, tau_target):
    """Budget at which a monotone curve tau(m) reaches tau_target, by linear
    interpolation in log m. Returns (m, censored): censored=True means the
    target is above the curve on the whole grid (value is a lower bound)."""
    logm = np.log(np.asarray(budgets, float))
    tau = np.maximum.accumulate(np.asarray(tau_curve, float))  # enforce monotonicity
    if tau_target <= tau[0]:
        return float(budgets[0]), False
    if tau_target > tau[-1]:
        return float(budgets[-1]), True
    k = int(np.searchsorted(tau, tau_target))
    frac = (tau_target - tau[k - 1]) / max(tau[k] - tau[k - 1], 1e-12)
    return float(np.exp(logm[k - 1] + frac * (logm[k] - logm[k - 1]))), False


def budget_saving(budgets, tau_bt, tau_hier, m0):
    """How many times more comparisons plain BT needs to match the quality
    the hierarchical model reaches with m0 comparisons per candidate."""
    target = tau_hier[list(budgets).index(m0)]
    m_bt, censored = equivalent_budget(budgets, tau_bt, target)
    return m_bt / m0, censored


# --- added after the first report ------------------------------------------------
def contrast_mse(u_true, scores, group):
    """MSE of within-group contrasts: both scores and true u are centred inside
    every group, the squared difference is averaged over candidates."""
    err = np.empty(len(u_true))
    for g in np.unique(group):
        m = group == g
        err[m] = (scores[m] - scores[m].mean()) - (u_true[m] - u_true[m].mean())
    return float(np.mean(err ** 2))


def top1(u_true, scores, groups):
    """Share of groups where the top-scored object is among the reference best
    (ties counted), and the chance level mean(|best| / |S|).
    `groups` is a list of index arrays (may overlap)."""
    hit, chance = [], []
    for S in groups:
        t = u_true[S]
        best = t == t.max()
        hit.append(float(best[int(np.argmax(scores[S]))]))
        chance.append(float(best.mean()))
    return float(np.mean(hit)), float(np.mean(chance))


def group_index_lists(group):
    return [np.flatnonzero(group == g) for g in np.unique(group)]


def tau_sets(u_true, scores, groups):
    """Mean Kendall tau-b over (possibly overlapping) groups given as index arrays.
    Groups with < 2 objects or constant reference are skipped; constant scores give 0."""
    taus = []
    for S in groups:
        if len(S) < 2 or np.all(u_true[S] == u_true[S][0]):
            continue
        t = kendalltau(u_true[S], scores[S]).statistic
        taus.append(0.0 if np.isnan(t) else t)
    return float(np.mean(taus)) if taus else np.nan
