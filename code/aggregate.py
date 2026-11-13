"""Aggregation over repeats for the added experiments: mean curves with bootstrap CI
and budget savings with CI (long-format tables: ..., method, metric, value)."""
import numpy as np
import pandas as pd

from metrics import budget_saving

BOOT_SEED = 20251113


def write_csv(df, path):
    """CSV with LF line endings on every platform (bit-identical reruns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def boot_index(n_rep, n_boot, salt=0):
    return np.random.default_rng([BOOT_SEED, salt]).integers(0, n_rep, (n_boot, n_rep))


def curves(raw, keys, rep_col="seed", n_boot=1000):
    """Mean, sd and 95% bootstrap CI of the mean over repeats for every `keys` combination."""
    out = []
    for key, sub in raw.groupby(keys, sort=True):
        v = sub.sort_values(rep_col)["value"].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        bm = v[boot_index(len(v), n_boot)].mean(axis=1)
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        lo, hi = np.percentile(bm, [2.5, 97.5])
        row.update(mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0, lo=lo, hi=hi, n=len(v))
        out.append(row)
    return pd.DataFrame(out)


def matrix(raw, method, metric, budgets, rep_col="seed"):
    sub = raw[(raw["method"] == method) & (raw["metric"] == metric)]
    piv = sub.pivot_table(index=rep_col, columns="budget", values="value", aggfunc="first")
    return piv.reindex(columns=budgets).sort_index().to_numpy(float)


def savings(raw, budgets, pairs, metrics, m0s, rep_col="seed", n_boot=1000, decreasing=()):
    """Saving of `target` vs `ref`: m_ref(quality of target at m0) / m0.
    Decreasing metrics (MSE, NLL) are negated. censored = ref does not reach the target
    on the budget grid (saving is then a lower bound); censored_boot = share of such
    bootstrap resamples."""
    rows = []
    for target, ref in pairs:
        for metric in metrics:
            sign = -1.0 if metric in decreasing else 1.0
            T = sign * matrix(raw, target, metric, budgets, rep_col)
            R = sign * matrix(raw, ref, metric, budgets, rep_col)
            if T.size == 0 or R.size == 0:
                continue
            idx = boot_index(T.shape[0], n_boot)
            Tb, Rb = T[idx].mean(axis=1), R[idx].mean(axis=1)
            for m0 in m0s:
                if m0 not in budgets:
                    continue
                s, cens = budget_saving(budgets, R.mean(0), T.mean(0), m0)
                bs = [budget_saving(budgets, Rb[b], Tb[b], m0) for b in range(n_boot)]
                vals = np.array([x[0] for x in bs])
                lo, hi = np.percentile(vals, [2.5, 97.5])
                rows.append(dict(target=target, ref=ref, metric=metric, m0=m0, saving=s,
                                 lo=lo, hi=hi, censored=bool(cens),
                                 censored_boot=float(np.mean([x[1] for x in bs]))))
    return pd.DataFrame(rows)
