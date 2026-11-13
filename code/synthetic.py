"""Synthetic tasks added after the first report (run in the process pool).

E1 additions are computed inside main.run_one (same lexicon and comparisons).
E2 (theory check) and E3 (survey designs) are a separate run with their own
random streams, independent of E1:
    lexicon      default_rng([seed, 20, rho_code])            (same lexicon for all designs)
    comparisons  default_rng([seed, 21, rho_code, design_id])
    lambda       default_rng([seed, 22, rho_code, int(m*100), design_id])
For design "random" (id 0) E3 uses exactly the E2 rows.
"""
import time

import numpy as np

import config as cfg
from data import make_lexicon, sample_comparisons
from metrics import contrast_mse, group_index_lists, top1, within_group_tau
from models import fit_scores, mean_nll


def rho_code(rho):
    return int(round(rho * 1000))


def extra_metrics(lex, scores, w, l, wc, lc, groups):
    """MSE of contrasts, top-1, NLL on training and on control comparisons."""
    t1, _ = top1(lex.u, scores, groups)
    return dict(mse=contrast_mse(lex.u, scores, lex.group), top1=t1,
                nll_train=mean_nll(scores, w, l), nll_ctrl=mean_nll(scores, wc, lc))


def control_comparisons(lex, seed, rho, design="random"):
    """N_CONTROL_SYNTH fresh comparisons, own stream [seed, 3, rho_code, design_id]."""
    rng = np.random.default_rng([seed, 3, rho_code(rho), cfg.DESIGN_IDS[design]])
    return sample_comparisons(rng, lex, 2 * cfg.N_CONTROL_SYNTH / len(lex.u), design)


def theory_task(task):
    """One (rho, seed, design) of E2/E3: tau and MSE for THEORY_METHODS on THEORY_BUDGETS."""
    rho, seed, design, budgets = task
    rc, did = rho_code(rho), cfg.DESIGN_IDS[design]
    lex = make_lexicon(np.random.default_rng([seed, 20, rc]), cfg.N_GROUPS, cfg.N_CANDIDATES,
                       cfg.N_FEATURES, rho, cfg.UTILITY_SCALE)
    w_all, l_all = sample_comparisons(np.random.default_rng([seed, 21, rc, did]), lex,
                                      max(budgets), design)
    n = len(lex.u)
    rows = []
    for m in budgets:
        n_cmp = max(1, int(round(m * n / 2)))
        w, l = w_all[:n_cmp], l_all[:n_cmp]
        for method in cfg.THEORY_METHODS:
            rng_fit = np.random.default_rng([seed, 22, rc, int(round(m * 100)), did])
            s, lam = fit_scores(method, lex.X, w, l, cfg, rng_fit, lex.prior_mean, rho)
            for metric, val in (("tau", within_group_tau(lex.u, s, lex.group)),
                                ("mse", contrast_mse(lex.u, s, lex.group)),
                                ("lambda", lam)):
                rows.append(dict(design=design, rho=rho, seed=seed, budget=m, method=method,
                                 metric=metric, value=val))
    return rows


def mc_task(task):
    """expected_risk_mc for one (design, mode) and all prior precisions a."""
    import theory
    design, mode, a, draws, grid_n = task
    grid = theory.mc_grid(cfg.MC_GRID_MIN, cfg.MC_GRID_MAX, grid_n)
    R = theory.expected_risk_mc(design, cfg.N_GROUPS, cfg.N_CANDIDATES, cfg.V, a, grid,
                                draws, mode, seed=cfg.DESIGN_IDS[design])
    return design, mode, grid, R


def runtime_task(task):
    """E7: median fitting time of every method for G groups (K = 8, m = RUNTIME_BUDGET)."""
    G, methods, repeats = task
    rho = 0.5
    lex = make_lexicon(np.random.default_rng([G, 70]), G, cfg.N_CANDIDATES, cfg.N_FEATURES,
                       rho, cfg.UTILITY_SCALE)
    w, l = sample_comparisons(np.random.default_rng([G, 71]), lex, cfg.RUNTIME_BUDGET)
    rows = []
    for method in methods:
        ts = []
        for r in range(repeats):
            t0 = time.perf_counter()
            fit_scores(method, lex.X, w, l, cfg, np.random.default_rng([G, 72, r]),
                       lex.prior_mean, rho)
            ts.append(time.perf_counter() - t0)
        rows.append(dict(G=G, N=len(lex.u), n_comparisons=len(w), method=method,
                         seconds=float(np.median(ts))))
    return rows


__all__ = ["extra_metrics", "control_comparisons", "theory_task", "mc_task", "runtime_task",
           "group_index_lists"]
