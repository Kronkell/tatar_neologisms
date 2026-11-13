"""E4/E5 tasks on real data (BenchLS groups, WCL annotators), run in the process pool.

Lambda is selected **by annotator**: inside the training subsample all comparisons of one random
annotator of group A are held out, lambda is chosen by the BT log-loss on them, then the model is
refitted on the whole subsample (the comparison-level 20 % hold-out of E1 is not used on real data:
comparisons of the same annotators are not independent).

Streams per repeat r: annotator split [r, 40]; pool permutation [r, 41]; control subsample [r, 42];
hold-out annotator for budget m [r, 46, int(m*100)]; old comparison hold-out (E5 path only) [r, 43, int(m*100)];
hold-out annotator for the full-pool theory fit [r, 47]; full-analogue subsamples [r, 45, draw].
"""
import numpy as np
from scipy.linalg import lapack

import config as cfg
from metrics import tau_sets, top1
from models import (_fit_bt, _fit_features, _fit_hierarchical, fit_fixed, fit_hierarchical_parts,
                    mean_nll)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def n_cmp(m, n):
    return max(1, int(round(m * n / 2)))


def neg_mean(R):
    """Utility = -mean complexity over the given annotator columns (NaN if none rated)."""
    cnt = np.sum(~np.isnan(R), axis=1)
    s = np.nansum(R, axis=1)
    out = np.full(len(R), np.nan)
    out[cnt > 0] = -s[cnt > 0] / cnt[cnt > 0]
    return out


def split(r, ctx):
    """Annotators A/B, permuted training pool, control subsample, reference and eval groups."""
    ann = np.random.default_rng([r, 40]).permutation(cfg.N_ANNOTATORS)
    A, B = np.sort(ann[:cfg.N_ANNOT_A]), np.sort(ann[cfg.N_ANNOT_A:])
    inA = np.isin(ctx["who"], A)
    pool = np.flatnonzero(inA)
    pool = pool[np.random.default_rng([r, 41]).permutation(len(pool))]
    ctrl = np.flatnonzero(~inA)
    ctrl = np.sort(ctrl[np.random.default_rng([r, 42]).permutation(len(ctrl))[:ctx["n_control"]]])
    refB = neg_mean(ctx["R"][:, B])
    refA = neg_mean(ctx["R"][:, A])
    eval_groups = []
    for S in ctx["groups"]:
        S = S[~np.isnan(refB[S])]
        if len(S) >= 2:
            eval_groups.append(S)
    both = [S[~np.isnan(refA[S])] for S in eval_groups]
    agreement = tau_sets(refB, np.nan_to_num(refA), [S for S in both if len(S) >= 2])
    return dict(A=A, B=B, pool=pool, ctrl=ctrl, refB=refB, eval_groups=eval_groups,
                agreement=agreement)


def evaluate(s, sp, ctx, w, l):
    t1, chance = top1(sp["refB"], s, sp["eval_groups"])
    W, L = ctx["w"], ctx["l"]
    return dict(tau=tau_sets(sp["refB"], s, sp["eval_groups"]), top1=t1, top1_chance=chance,
                nll_train=mean_nll(s, w, l),
                nll_ctrl=mean_nll(s, W[sp["ctrl"]], L[sp["ctrl"]]))


def holdout_annotator(r, m, A):
    return int(np.random.default_rng([r, 46, int(round(m * 100))]).choice(A))


def select_lambda_annotator(fit, w, l, who, a, grid):
    """lambda with the best BT log-loss on all comparisons of annotator a (fitted on the rest)."""
    hold = who == a
    if hold.sum() < 2 or (~hold).sum() < 2:
        return grid[len(grid) // 2], [np.nan] * len(grid)
    losses = [mean_nll(fit(w[~hold], l[~hold], lam), w[hold], l[hold]) for lam in grid]
    return grid[int(np.argmin(losses))], losses


def fit_method(method, X, w, l, who, a):
    """(scores, lambda) for bt / features / hierarchical with lambda chosen by annotator a."""
    n = X.shape[0]
    if method == "features":
        if X.shape[1] == 0:
            return np.zeros(n), np.nan
        return _fit_features(X, w, l, cfg.LAMBDA_THETA), np.nan
    if method == "bt" or X.shape[1] == 0:
        fit = lambda ww, ll, lam: _fit_bt(n, ww, ll, lam)
    elif method == "hierarchical":
        fit = lambda ww, ll, lam: _fit_hierarchical(X, ww, ll, lam, cfg.LAMBDA_THETA)
    else:
        raise ValueError(method)
    lam, _ = select_lambda_annotator(fit, w, l, who, a, cfg.LAMBDA_GRID)
    return fit(w, l, lam), lam


def subsample(sp, ctx, m):
    N = len(ctx["R"])
    k = n_cmp(m, N)
    capped = k > len(sp["pool"])
    idx = sp["pool"][:min(k, len(sp["pool"]))]
    return ctx["w"][idx], ctx["l"][idx], ctx["who"][idx], len(idx), capped


def real_task(task):
    """E4 for one (repeat, budget): bt once, features and hierarchical for every feature set."""
    r, m, ctx, sets = task
    sp = split(r, ctx)
    w, l, who, k, capped = subsample(sp, ctx, m)
    a = holdout_annotator(r, m, sp["A"])
    rows = []

    def add(fs, method, s, lam):
        vals = dict(lambda_=lam, n_comparisons=k, capped=float(capped), holdout_annotator=a,
                    **evaluate(s, sp, ctx, w, l))
        for metric, v in vals.items():
            rows.append(dict(seed=r, budget=m, feature_set=fs, method=method, metric=metric,
                             value=float(v)))

    s, lam = fit_method("bt", ctx["X"], w, l, who, a)
    add("none", "bt", s, lam)
    for fs, cols in sets.items():
        X = ctx["X"][:, cols]
        for method in ("features", "hierarchical"):
            s, lam = fit_method(method, X, w, l, who, a)
            add(fs, method, s, lam)
    info = dict(seed=r, A=sp["A"].tolist(), B=sp["B"].tolist(), pool=len(sp["pool"]),
                agreement=sp["agreement"], n_eval_groups=len(sp["eval_groups"]))
    return rows, info


def complexity_task(task):
    """E5: hierarchical / features with the first d features (lambda by annotator), and the
    lambda path of hierarchical at m = LAMBDA_PATH_BUDGET with both hold-out rules."""
    r, ctx, budgets, lam_path = task
    sp = split(r, ctx)
    X4 = ctx["X"]
    rows, path = [], []
    for m in budgets:
        w, l, who, _, _ = subsample(sp, ctx, m)
        a = holdout_annotator(r, m, sp["A"])
        for d in range(X4.shape[1] + 1):
            for method in ("hierarchical", "features"):
                s, lam = fit_method(method, X4[:, :d], w, l, who, a)
                for metric, v in dict(lambda_=lam, **evaluate(s, sp, ctx, w, l)).items():
                    rows.append(dict(seed=r, budget=m, method=method, d=d, metric=metric,
                                     value=float(v)))
    m = cfg.LAMBDA_PATH_BUDGET
    w, l, who, _, _ = subsample(sp, ctx, m)
    a = holdout_annotator(r, m, sp["A"])
    hold = who == a
    rng = np.random.default_rng([r, 43, int(round(m * 100))])      # old rule, for comparison
    n_val = int(round(cfg.HOLDOUT_FRACTION * len(w)))
    perm = rng.permutation(len(w))
    val, tr = perm[:n_val], perm[n_val:]
    for lam in lam_path:
        s = fit_fixed("hierarchical", X4, w, l, lam, cfg.LAMBDA_THETA)
        s_ann = fit_fixed("hierarchical", X4, w[~hold], l[~hold], lam, cfg.LAMBDA_THETA)
        s_cmp = fit_fixed("hierarchical", X4, w[tr], l[tr], lam, cfg.LAMBDA_THETA)
        ev = evaluate(s, sp, ctx, w, l)
        path.append(dict(seed=r, lam=lam, tau=ev["tau"], top1=ev["top1"],
                         nll_train=ev["nll_train"], nll_ctrl=ev["nll_ctrl"],
                         nll_holdout_annotator=mean_nll(s_ann, w[hold], l[hold]),
                         nll_holdout_comparisons=mean_nll(s_cmp, w[val], l[val])))
    return rows, path


def theory_inputs(r, ctx, cols):
    """Fit hierarchical (features `cols`) on the full pool A, lambda by annotator:
    û, V̂, ρ̂, pairs with q_e and Ī_e."""
    sp = split(r, ctx)
    X = ctx["X"][:, cols]
    pool = sp["pool"]
    w, l, who = ctx["w"][pool], ctx["l"][pool], ctx["who"][pool]
    a = int(np.random.default_rng([r, 47]).choice(sp["A"]))
    fit = lambda ww, ll, lam: _fit_hierarchical(X, ww, ll, lam, cfg.LAMBDA_THETA)
    lam, _ = select_lambda_annotator(fit, w, l, who, a, cfg.LAMBDA_GRID)
    theta, eta = fit_hierarchical_parts(X, w, l, lam, cfg.LAMBDA_THETA)
    u = X @ theta + eta
    V = float(np.var(u))
    var_f = float(np.var(X @ theta))
    rho = var_f / V
    i, j = np.minimum(w, l), np.maximum(w, l)
    key = i.astype(np.int64) * len(u) + j
    uniq, cnt = np.unique(key, return_counts=True)
    pi, pj = uniq // len(u), uniq % len(u)
    p = sigmoid(u[pi] - u[pj])
    Ie = p * (1 - p)
    q = cnt / cnt.sum()
    pc = sigmoid(u[w] - u[l])
    # diagnostic alternative: residual variance from the prior 1/lambda instead of Var(eta-hat)
    V_prior = var_f + 1.0 / lam
    return dict(lam=lam, V=V, rho=rho, V_prior=V_prior, rho_prior=var_f / V_prior, pairs=np.column_stack([pi, pj]), q=q, Ie=Ie,
                Ibar=float(np.sum(q * Ie)), info_per_cmp=pc * (1 - pc), pool_i=i, pool_j=j,
                pool_size=len(w))


def real_theory_task(task):
    """Mean-field prediction (equivalent_budget_theory, spectral_bounds) for one repeat and set.
    For keep_inputs=True also returns the inputs of the 'full' analogue."""
    import theory
    r, ctx, fs, cols, m0s, keep_inputs = task
    ti = theory_inputs(r, ctx, cols)
    N = len(ctx["R"])
    L = theory.laplacian_from_pairs(ti["pairs"], ti["q"], ti["Ie"], N)
    groups = ctx["groups"]
    out = dict(seed=r, feature_set=fs, lam=ti["lam"], V=ti["V"], rho=ti["rho"], Ibar=ti["Ibar"],
               V_prior=ti["V_prior"], rho_prior=ti["rho_prior"], pool=ti["pool_size"])
    for m0 in m0s:
        m_bt = theory.equivalent_budget_theory(L, groups, ti["rho"], ti["V"], m0)
        lo, hi = theory.spectral_bounds(L, groups, ti["rho"], ti["V"], m0)
        out[f"spectral_m0={m0:g}"] = m_bt / m0
        out[f"bound_lo_m0={m0:g}"] = lo / m0
        out[f"bound_hi_m0={m0:g}"] = hi / m0
        m_alt = theory.equivalent_budget_theory(L, groups, ti["rho_prior"], ti["V_prior"], m0)
        out[f"spectral_prior_m0={m0:g}"] = m_alt / m0
    return out, (ti if keep_inputs else None)


def _risk_dense(Lmat, a, groups, N):
    """Σ_g tr(C Σ_SS) / N with Σ = (L + aI)⁻¹. Single precision Cholesky inverse:
    2× faster than double, relative error of the risk ≤ 1e-7 on these matrices."""
    M = (Lmat + a * np.eye(N)).astype(np.float32)
    c, info = lapack.spotrf(M, lower=1)
    if info != 0:
        raise np.linalg.LinAlgError(info)
    inv, info = lapack.spotri(c, lower=1)
    inv = np.tril(inv).astype(np.float64)
    inv = inv + inv.T - np.diag(np.diag(inv))
    tot = 0.0
    for S in groups:
        B = inv[np.ix_(S, S)]
        tot += np.trace(B) - B.sum() / len(S)
    return tot / N


def real_full_task(task):
    """'full' analogue on real data: risk averaged over random subsamples of pool A
    (prefixes of a random permutation), information p̂(1−p̂) of every comparison by û."""
    r, fs, draws, ti, groups, N, grid_bt, m0s = task
    a_bt = 1.0 / ti["V"]
    a_h = 1.0 / ((1.0 - ti["rho"]) * ti["V"])
    i, j, info = ti["pool_i"], ti["pool_j"], ti["info_per_cmp"]
    pts = sorted({("bt", m) for m in grid_bt} | {("h", m) for m in m0s}, key=lambda x: (x[1], x[0]))
    res = {p: 0.0 for p in pts}
    for dr in draws:
        perm = np.random.default_rng([r, 45, dr]).permutation(len(i))
        Lm = np.zeros((N, N))
        done = 0
        for kind, m in pts:
            k = min(n_cmp(m, N), len(i))
            if k > done:
                sl = perm[done:k]
                ii, jj, ww = i[sl], j[sl], info[sl]
                np.add.at(Lm, (ii, ii), ww)
                np.add.at(Lm, (jj, jj), ww)
                np.add.at(Lm, (ii, jj), -ww)
                np.add.at(Lm, (jj, ii), -ww)
                done = k
            res[(kind, m)] += _risk_dense(Lm, a_bt if kind == "bt" else a_h, groups, N)
    return fs, res


def saving_from_risk(grid_bt, risk_bt, risk_h_m0, m0):
    """m_BT/m0 where the BT risk curve (log–log interpolation) reaches risk_h_m0; inf if not."""
    lm = np.log(np.asarray(grid_bt, float))
    lr = np.log(np.minimum.accumulate(np.asarray(risk_bt, float)))
    t = np.log(risk_h_m0)
    if lr[-1] > t:
        return np.inf
    k = int(np.argmax(lr <= t))
    if k == 0:
        return float(np.exp(lm[0])) / m0
    f = (lr[k - 1] - t) / (lr[k - 1] - lr[k])
    return float(np.exp(lm[k - 1] + f * (lm[k] - lm[k - 1]))) / m0
