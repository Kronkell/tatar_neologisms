"""Single entry point of the computational experiment.

    python code/main.py            # full experiment  -> results/
    python code/main.py --quick    # small smoke run  -> results/quick/
    python code/main.py --only {synthetic,designs,real,llm,runtime}

Computes all numbers and saves them; figures are drawn separately by plots.py
from the saved files, so plots can be restyled without re-running this.
"""
import argparse
import json
import os
import time
from itertools import product
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import config as cfg
from data import make_lexicon, sample_comparisons
from metrics import budget_saving, group_index_lists, within_group_tau
from models import fit_scores
import experiments
import synthetic


def run_one(task):
    """All budgets and methods for one (rho, seed). Comparisons for smaller
    budgets are prefixes of the largest-budget sample, so curves are nested."""
    rho, seed, budgets = task
    lex = make_lexicon(np.random.default_rng(seed), cfg.N_GROUPS, cfg.N_CANDIDATES,
                       cfg.N_FEATURES, rho, cfg.UTILITY_SCALE)
    rng_cmp = np.random.default_rng([seed, 1, int(round(rho * 1000))])
    w_all, l_all = sample_comparisons(rng_cmp, lex, max(budgets))
    n = len(lex.u)
    # added: control comparisons from an own stream (old streams untouched)
    wc, lc = synthetic.control_comparisons(lex, seed, rho)
    groups = group_index_lists(lex.group)
    rows, ext = [], []
    for m in budgets:
        n_cmp = max(1, int(round(m * n / 2)))
        w, l = w_all[:n_cmp], l_all[:n_cmp]
        for method in cfg.METHODS + cfg.NEW_METHODS:
            rng_fit = np.random.default_rng([seed, 2, int(round(rho * 1000)), int(m * 100)])
            scores, lam = fit_scores(method, lex.X, w, l, cfg, rng_fit, lex.prior_mean, rho)
            tau = within_group_tau(lex.u, scores, lex.group)
            if method in cfg.METHODS:   # original E1 rows, unchanged
                rows.append(dict(rho=rho, budget=m, seed=seed, method=method,
                                 n_comparisons=n_cmp, lambda_=lam, tau=tau))
            extra = dict(lambda_=lam, tau=tau,
                         **synthetic.extra_metrics(lex, scores, w, l, wc, lc, groups))
            for metric, val in extra.items():
                ext.append(dict(rho=rho, budget=m, seed=seed, method=method,
                                metric=metric, value=val))
    return rows, ext


def bootstrap_rows(raw, budgets, rng):
    """Bootstrap over seeds: mean curves with 95% CI and budget savings."""
    curves, savings = [], []
    for rho, df in raw.groupby("rho"):
        piv = df.pivot_table(index="seed", columns=["method", "budget"], values="tau")
        seeds = piv.index.to_numpy()
        mean = {meth: piv[meth][budgets].mean().to_numpy() for meth in cfg.METHODS}
        boot_idx = rng.integers(0, len(seeds), (cfg.N_BOOTSTRAP, len(seeds)))
        boot = {meth: np.stack([piv[meth][budgets].to_numpy()[ix].mean(0) for ix in boot_idx])
                for meth in cfg.METHODS}
        for meth in cfg.METHODS:
            lo, hi = np.percentile(boot[meth], [2.5, 97.5], axis=0)
            for k, m in enumerate(budgets):
                curves.append(dict(rho=rho, budget=m, method=meth, tau_mean=mean[meth][k],
                                   tau_lo=lo[k], tau_hi=hi[k]))
        for m0 in [b for b in cfg.REFERENCE_BUDGETS if b in budgets]:
            point, censored = budget_saving(budgets, mean["bt"], mean["hierarchical"], m0)
            bs = [budget_saving(budgets, boot["bt"][b], boot["hierarchical"][b], m0)[0]
                  for b in range(cfg.N_BOOTSTRAP)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            savings.append(dict(rho=rho, m0=m0, saving=point, saving_lo=lo, saving_hi=hi,
                                censored=censored))
    return pd.DataFrame(curves), pd.DataFrame(savings)


def run_e1(args):
    """Original E1 (results/raw_results.csv, curves.csv, savings.csv, summary.json)
    plus additions (results/e1_raw.csv, e1_curves.csv, e1_savings.csv)."""
    rhos, budgets, seeds = cfg.RHOS, cfg.BUDGETS, cfg.SEEDS
    out_dir = cfg.RESULTS_DIR
    if args.quick:
        rhos, budgets, seeds = [0.5, 0.9], [1, 2, 4], [0, 1, 2]
        out_dir = cfg.RESULTS_DIR / "quick"
        cfg.N_BOOTSTRAP = 100
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tasks = [(rho, seed, budgets) for rho, seed in product(rhos, seeds)]
    with get_context("spawn").Pool(cfg.N_JOBS) as pool:
        parts = list(pool.imap_unordered(run_one, tasks))
    rows = [r for chunk, _ in parts for r in chunk]
    ext = [r for _, chunk in parts for r in chunk]
    raw = (pd.DataFrame(rows)
           .sort_values(["rho", "budget", "seed", "method"]).reset_index(drop=True))
    curves, savings = bootstrap_rows(raw, budgets, np.random.default_rng(cfg.BOOTSTRAP_SEED))

    # LF line endings: the archived results were written on a POSIX system
    raw.to_csv(out_dir / "raw_results.csv", index=False, lineterminator="\n")
    curves.to_csv(out_dir / "curves.csv", index=False, lineterminator="\n")
    savings.to_csv(out_dir / "savings.csv", index=False, lineterminator="\n")
    summary = dict(
        runtime_sec=round(time.time() - t0, 1),
        n_groups=cfg.N_GROUPS, n_candidates=cfg.N_CANDIDATES, n_features=cfg.N_FEATURES,
        utility_scale=cfg.UTILITY_SCALE, rhos=rhos, budgets=budgets, n_seeds=len(seeds),
        lambda_grid=cfg.LAMBDA_GRID, main_rho=cfg.MAIN_RHO,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), newline="\n")
    experiments.e1_additions(pd.DataFrame(ext), budgets, out_dir, cfg.N_BOOTSTRAP)

    print(curves.round(3).to_string(index=False))
    print(savings.round(2).to_string(index=False))
    print(f"saved to {out_dir} in {summary['runtime_sec']} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    parser.add_argument("--only", choices=["synthetic", "designs", "real", "llm", "runtime"],
                        help="run one experiment group (default: all)")
    args = parser.parse_args()
    timing = {}

    def step(name, fn):
        if args.only and args.only != name.split(":")[0]:
            return
        t = time.time()
        fn()
        timing[name] = round(time.time() - t, 1)
        print(f"[{name}] {timing[name]} s", flush=True)

    step("synthetic:E1", lambda: run_e1(args))
    step("synthetic:E2", lambda: experiments.run_theory(args))
    step("designs:E3", lambda: experiments.run_designs(args))
    step("real:E4", lambda: experiments.run_real(args))
    step("real:E5", lambda: experiments.run_real_analysis(args))
    step("llm:E8", lambda: experiments.run_llm(args))
    step("runtime:E7", lambda: experiments.run_runtime(args))
    experiments.close_pool()
    experiments.make_tables(args)   # E6: from whatever results exist
    experiments.design_mse_vs_risk(args)   # derived from saved E3 results
    experiments.save_timing(args, timing)


if __name__ == "__main__":
    main()
