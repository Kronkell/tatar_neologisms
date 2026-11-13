"""Experiments added after the first report: E1 additions, E2–E8 (called from main.py)."""
import json
import time
from itertools import product
from multiprocessing import get_context

import numpy as np
import pandas as pd

import config as cfg
from aggregate import curves, savings, write_csv

DECREASING = ("mse", "nll_train", "nll_ctrl")


def out_dir(args):
    d = cfg.RESULTS_DIR / "quick" if args.quick else cfg.RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


_POOL = []


def pool_map(fn, tasks):
    """One shared spawn pool for all added experiments (closed by close_pool)."""
    if not _POOL:
        _POOL.append(get_context("spawn").Pool(cfg.N_JOBS))
    return list(_POOL[0].imap(fn, tasks))


def close_pool():
    if _POOL:
        _POOL.pop().close()


def save_timing(args, timing):
    path = out_dir(args) / "timing.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(timing)
    path.write_text(json.dumps(old, indent=2), newline="\n")


# ----------------------------------------------------------------------------- E1+
def e1_additions(ext, budgets, out, n_boot):
    """bt_fixed, hier_oracle, MSE / top-1 / NLL for E1 (same lexicons and comparisons)."""
    ext = ext.sort_values(["rho", "budget", "seed", "method", "metric"], kind="mergesort")
    write_csv(ext.reset_index(drop=True), out / "e1_raw.csv")
    write_csv(curves(ext, ["rho", "budget", "method", "metric"], n_boot=n_boot),
              out / "e1_curves.csv")
    rows = []
    for rho, sub in ext.groupby("rho"):
        for pairs, metrics in (([("hierarchical", "bt"), ("hier_oracle", "bt_fixed")],
                                ["tau", "mse"]),):
            s = savings(sub, budgets, pairs, metrics, [b for b in cfg.REFERENCE_BUDGETS
                                                       if b in budgets],
                        n_boot=n_boot, decreasing=DECREASING)
            s.insert(0, "rho", rho)
            rows.append(s)
    write_csv(pd.concat(rows, ignore_index=True), out / "e1_savings.csv")


# ----------------------------------------------------------------------------- E2/E3
def synth_settings(args):
    if args.quick:
        return dict(rhos=[0.5], budgets=[0.5, 1, 2, 4, 8], seeds=[0, 1, 2], draws=10, grid_n=30,
                    n_boot=100, design_rhos=[0.5])
    return dict(rhos=cfg.RHOS, budgets=cfg.THEORY_BUDGETS, seeds=cfg.THEORY_SEEDS,
                draws=cfg.MC_DRAWS, grid_n=cfg.MC_GRID_N, n_boot=cfg.N_BOOTSTRAP,
                design_rhos=cfg.DESIGN_RHOS)


def _theory_raw(designs, rhos, st):
    import synthetic
    tasks = [(rho, seed, d, st["budgets"]) for d in designs for rho in rhos for seed in st["seeds"]]
    raw = pd.DataFrame([r for rows in pool_map(synthetic.theory_task, tasks) for r in rows])
    return raw.sort_values(["design", "rho", "budget", "seed", "method", "metric"],
                           kind="mergesort").reset_index(drop=True)


def _mc(designs, rhos, st):
    """E R(m, a)/N on MC_GRID for a_BT and every a_H; modes mean_field (analytic), design, full."""
    import synthetic
    import theory
    a = np.array([1.0 / cfg.V] + [1.0 / ((1.0 - r) * cfg.V) for r in rhos])
    labels = ["bt"] + [f"h{r:g}" for r in rhos]
    grid = theory.mc_grid(cfg.MC_GRID_MIN, cfg.MC_GRID_MAX, st["grid_n"])
    out = {}
    for d in designs:
        out[(d, "mean_field")] = theory.expected_risk_mc(d, cfg.N_GROUPS, cfg.N_CANDIDATES, cfg.V,
                                                         a, grid, mode="mean_field")
    tasks = [(d, mode, a, st["draws"], st["grid_n"]) for d in designs for mode in ("design", "full")]
    for d, mode, _, R in pool_map(synthetic.mc_task, tasks):
        out[(d, mode)] = R
    rows = [dict(design=d, mode=mode, prior=labels[k], a=a[k], m=m, risk=R[k, t])
            for (d, mode), R in out.items() for k in range(len(a)) for t, m in enumerate(grid)]
    return out, grid, labels, pd.DataFrame(rows)


def _predictions(design, rho, m0, mc, grid, labels):
    import theory
    G, K, V = cfg.N_GROUPS, cfg.N_CANDIDATES, cfg.V
    groups = [np.arange(g * K, (g + 1) * K) for g in range(G)]
    L = theory.design_laplacian(design, groups, theory.mean_pq(V))
    m_bt = theory.equivalent_budget_theory(L, groups, rho, V, m0)
    lo, hi = theory.spectral_bounds(L, groups, rho, V, m0)
    row = dict(pred_spectral=m_bt / m0, bound_lo=lo / m0, bound_hi=hi / m0,
               inside_bounds=bool(lo - 1e-9 <= m_bt <= hi + 1e-9))
    k = labels.index(f"h{rho:g}")
    for mode in ("mean_field", "design", "full"):
        R = mc[(design, mode)]
        row[f"mc_{mode}"] = theory.equivalent_budget_curves(grid, R[0], R[k], m0)
    return row


def _measured(raw, budgets, n_boot, m0s):
    return savings(raw, budgets, [("hierarchical", "bt"), ("hier_oracle", "bt_fixed")],
                   ["tau", "mse"], m0s, n_boot=n_boot, decreasing=DECREASING)


MEAS = {"hier_tau": ("hierarchical", "tau"), "hier_mse": ("hierarchical", "mse"),
        "oracle_tau": ("hier_oracle", "tau"), "oracle_mse": ("hier_oracle", "mse")}


def _check_table(raw, design, rhos, st, mc, grid, labels, with_closed):
    import theory
    rows = []
    for rho in rhos:
        sub = raw[(raw.design == design) & (raw.rho == rho)]
        meas = _measured(sub, st["budgets"], st["n_boot"],
                         [m for m in cfg.REFERENCE_BUDGETS if m in st["budgets"]])
        for m0 in sorted(meas.m0.unique()):
            row = dict(design=design, rho=rho, m0=m0)
            if with_closed:
                row["pred_closed"] = theory.saving_complete(rho, m0, cfg.N_CANDIDATES, cfg.V)
            row.update(_predictions(design, rho, m0, mc, grid, labels))
            for name, (target, metric) in MEAS.items():
                r = meas[(meas.m0 == m0) & (meas.target == target) & (meas.metric == metric)].iloc[0]
                row.update({f"meas_{name}": r.saving, f"meas_{name}_lo": r.lo,
                            f"meas_{name}_hi": r.hi, f"meas_{name}_censored": r.censored,
                            f"meas_{name}_censored_boot": r.censored_boot})
            # decomposition (MSE, oracles): mean_field -> design -> full -> measured
            row["dec_design_effect"] = row["mc_design"] / row["mc_mean_field"]
            row["dec_info_effect"] = row["mc_full"] / row["mc_design"]
            row["dec_residual"] = row["meas_oracle_mse"] / row["mc_full"]
            row["dec_estimation_cost"] = row["meas_hier_mse"] / row["meas_oracle_mse"]
            row["dec_mse_to_tau"] = row["meas_hier_tau"] / row["meas_hier_mse"]
            rows.append(row)
    return pd.DataFrame(rows)


def run_theory(args):
    """E2: separate run (THEORY_BUDGETS, 30 seeds, own streams), plan random."""
    out, st = out_dir(args), synth_settings(args)
    raw = _theory_raw(["random"], st["rhos"], st)
    write_csv(raw, out / "theory_raw.csv")
    write_csv(curves(raw, ["design", "rho", "budget", "method", "metric"], n_boot=st["n_boot"]),
              out / "theory_curves.csv")
    mc, grid, labels, mc_rows = _mc(["random"], st["rhos"], st)
    write_csv(mc_rows, out / "theory_mc_risk.csv")
    write_csv(_check_table(raw, "random", st["rhos"], st, mc, grid, labels, True),
              out / "theory_check.csv")


def run_designs(args):
    """E3: cycle and star (random is taken from E2: same streams and settings)."""
    out, st = out_dir(args), synth_settings(args)
    rhos = st["design_rhos"]
    prev = out / "theory_raw.csv"
    raw_r = pd.read_csv(prev) if prev.exists() else _theory_raw(["random"], rhos, st)
    raw_r = raw_r[raw_r.rho.isin(rhos)]
    raw = pd.concat([raw_r, _theory_raw(["cycle", "star"], rhos, st)], ignore_index=True)
    raw = raw.sort_values(["design", "rho", "budget", "seed", "method", "metric"],
                          kind="mergesort").reset_index(drop=True)
    write_csv(raw, out / "designs_raw.csv")
    mc, grid, labels, mc_rows = _mc(cfg.DESIGNS, rhos, st)
    write_csv(mc_rows, out / "designs_mc_risk.csv")
    tab = pd.concat([_check_table(raw, d, rhos, st, mc, grid, labels, False) for d in cfg.DESIGNS],
                    ignore_index=True)
    write_csv(tab, out / "designs_savings.csv")


# ----------------------------------------------------------------------------- E4/E5
_REAL = {}


def real_settings(args):
    if args.quick:
        return dict(seeds=[0], budgets=[0.5, 1, 2], m0s=[0.5, 1], n_control=2000, draws=0,
                    n_boot=100, sets={k: cfg.REAL_FEATURE_SETS[k] for k in ("zipf", "all")},
                    complexity_budgets=[2], lambda_path=[0.1, 1.0, 10.0])
    return dict(seeds=cfg.REAL_SEEDS, budgets=cfg.REAL_BUDGETS, m0s=cfg.REAL_M0S,
                n_control=cfg.N_CONTROL_REAL, draws=cfg.REAL_MC_DRAWS, n_boot=cfg.N_BOOTSTRAP,
                sets=cfg.REAL_FEATURE_SETS, complexity_budgets=cfg.COMPLEXITY_BUDGETS,
                lambda_path=cfg.LAMBDA_PATH)


def real_context(args):
    """Data, control counts (stop if they differ), features and all comparisons."""
    if "ctx" in _REAL:
        return _REAL["ctx"], _REAL["meta"]
    import real_data
    data = real_data.load_real()
    counts = real_data.control_counts(data)
    if counts != cfg.EXPECTED_COUNTS:
        raise SystemExit(f"control counts differ: {counts} != {cfg.EXPECTED_COUNTS}")
    X = real_data.feature_matrix(data.words)
    (w, l, who), _ = real_data.all_comparisons(data)
    ctx = dict(X=X, R=data.ratings, groups=data.groups, w=w, l=l, who=who,
               n_control=real_settings(args)["n_control"])
    meta = dict(control_counts=counts, expected_counts=cfg.EXPECTED_COUNTS,
                wcl_lowercase_duplicates=data.n_wcl_dups, features=cfg.REAL_FEATURES,
                feature_corr=np.corrcoef(X.T).round(3).tolist())
    _REAL["ctx"], _REAL["meta"] = ctx, meta
    return ctx, meta


def _mean_sd(x):
    x = np.asarray(x, float)
    return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)


def run_real(args):
    """E4: real-data protocol for every feature set (lambda by annotator), measured savings
    and theory predictions (mean field on all splits, 'full' analogue on split 0)."""
    import real_exp
    t0 = time.time()
    out, st = out_dir(args), real_settings(args)
    ctx, meta = real_context(args)
    N = len(ctx["R"])
    sets, budgets, m0s = st["sets"], st["budgets"], st["m0s"]

    res = pool_map(real_exp.real_task, [(r, m, ctx, sets) for r in st["seeds"] for m in budgets])
    raw = pd.DataFrame([row for rows, _ in res for row in rows])
    raw = raw.sort_values(["feature_set", "seed", "budget", "method", "metric"], kind="mergesort")
    write_csv(raw.reset_index(drop=True), out / "real_raw.csv")
    info = {i["seed"]: i for _, i in res}
    write_csv(curves(raw, ["feature_set", "budget", "method", "metric"], n_boot=st["n_boot"]),
              out / "real_curves.csv")

    # theory: mean field for every split and set; inputs of the 'full' analogue for split 0
    s0 = st["seeds"][0]
    th_res = pool_map(real_exp.real_theory_task,
                      [(r, ctx, fs, cols, m0s, r == s0) for fs, cols in sets.items()
                       for r in st["seeds"]])
    th = pd.DataFrame([o for o, _ in th_res])
    write_csv(th, out / "real_theory.csv")
    full, risk_rows, grids = {}, [], {}
    if st["draws"] > 0:
        tis = {o["feature_set"]: ti for o, ti in th_res if ti is not None}
        n_chunks = 10
        tasks = []
        for fs, ti in tis.items():
            m_pool = 2 * ti["pool_size"] / N
            grid = sorted(set([b for b in budgets if b <= m_pool] + [m_pool]))
            grids[fs] = grid
            for c in range(n_chunks):
                tasks.append((s0, fs, list(range(c, st["draws"], n_chunks)), ti, ctx["groups"], N,
                              grid, m0s))
        tot = {}
        for fs, part in pool_map(real_exp.real_full_task, tasks):
            acc = tot.setdefault(fs, {})
            for k, v in part.items():
                acc[k] = acc.get(k, 0.0) + v / st["draws"]
        for fs, acc in tot.items():
            g = grids[fs]
            full[fs] = {m0: real_exp.saving_from_risk(g, [acc[("bt", m)] for m in g],
                                                      acc[("h", m0)], m0) for m0 in m0s}
            risk_rows += [dict(feature_set=fs, kind=k, m=m, risk=v) for (k, m), v in sorted(acc.items())]
        write_csv(pd.DataFrame(risk_rows), out / "real_full_risk.csv")

    rows = []
    bt_rows = raw[raw.method == "bt"]
    for fs in sets:
        sub = pd.concat([raw[raw.feature_set == fs], bt_rows], ignore_index=True)
        meas = savings(sub, budgets, [("hierarchical", "bt"), ("features", "bt")],
                       ["tau", "nll_ctrl"], m0s, n_boot=st["n_boot"], decreasing=DECREASING)
        t = th[th.feature_set == fs]
        for rr in meas.itertuples(index=False):
            row = dict(feature_set=fs, **rr._asdict())
            if rr.target == "hierarchical":
                k = f"m0={rr.m0:g}"
                row.update(pred_mean_field=t[f"spectral_{k}"].mean(),
                           pred_mean_field_sd=t[f"spectral_{k}"].std(ddof=1),
                           pred_mean_field_seed0=t[t.seed == s0][f"spectral_{k}"].iloc[0],
                           pred_mean_field_prior_var=t[f"spectral_prior_{k}"].mean(),
                           bound_lo=t[f"bound_lo_{k}"].mean(),
                           bound_hi_finite_splits=int(np.isfinite(t[f"bound_hi_{k}"]).sum()),
                           pred_full_seed0=full.get(fs, {}).get(rr.m0, np.nan))
            rows.append(row)
    write_csv(pd.DataFrame(rows), out / "real_savings.csv")

    meta = dict(meta)
    meta.update(
        N=N, n_groups=len(ctx["groups"]), seeds=st["seeds"], budgets=budgets, m0s=m0s,
        feature_sets={k: [cfg.REAL_FEATURES[c] for c in v] for k, v in sets.items()},
        lambda_selection="annotator: all comparisons of one random A annotator are held out",
        pool_size=_mean_sd([i["pool"] for i in info.values()]),
        pool_min=int(min(i["pool"] for i in info.values())),
        capped_cells=int(raw[(raw.metric == "capped") & (raw.value > 0)].shape[0]),
        n_eval_groups=_mean_sd([i["n_eval_groups"] for i in info.values()]),
        agreement_A_B_tau=_mean_sd([i["agreement"] for i in info.values()]),
        agreement_per_seed=[info[r]["agreement"] for r in st["seeds"]],
        top1_chance=float(raw[(raw.metric == "top1_chance")].value.mean()),
        per_set={fs: dict(rho_hat=_mean_sd(th[th.feature_set == fs].rho),
                          V_hat=_mean_sd(th[th.feature_set == fs].V),
                          Ibar_hat=_mean_sd(th[th.feature_set == fs].Ibar),
                          rho_hat_prior_var=_mean_sd(th[th.feature_set == fs].rho_prior),
                          V_hat_prior_var=_mean_sd(th[th.feature_set == fs].V_prior),
                          lambda_full_pool=th[th.feature_set == fs].lam.tolist())
                 for fs in sets},
        full_analogue=dict(seed=s0, draws=st["draws"], grids=grids,
                           saving={fs: {f"m0={k:g}": v for k, v in d.items()}
                                   for fs, d in full.items()},
                           precision="float32 Cholesky inverse (relative risk error <= 1e-7)"),
        splits=[dict(seed=r, A=info[r]["A"], B=info[r]["B"]) for r in st["seeds"]],
        runtime_sec=round(time.time() - t0, 1),
    )
    (out / "real_dataset.json").write_text(json.dumps(meta, indent=2), newline="\n")


def run_real_analysis(args):
    """E5: model complexity (first d features) and the lambda path (both hold-out rules)."""
    import real_exp
    out, st = out_dir(args), real_settings(args)
    ctx, _ = real_context(args)
    res = pool_map(real_exp.complexity_task,
                   [(r, ctx, st["complexity_budgets"], st["lambda_path"]) for r in st["seeds"]])
    comp = pd.DataFrame([row for rows, _ in res for row in rows])
    comp = comp.sort_values(["seed", "budget", "method", "d", "metric"], kind="mergesort")
    write_csv(comp.reset_index(drop=True), out / "real_complexity.csv")
    path = pd.DataFrame([row for _, rows in res for row in rows])
    g = path.groupby("seed")
    path["selected_by_annotator"] = g.nll_holdout_annotator.transform("min") == path.nll_holdout_annotator
    path["selected_by_comparisons"] = (g.nll_holdout_comparisons.transform("min")
                                       == path.nll_holdout_comparisons)
    write_csv(path.sort_values(["seed", "lam"]).reset_index(drop=True), out / "real_lambda_path.csv")


def run_llm(args):
    """E8 reads only the cache data/llm/ written by code/collect_llm.py."""
    if not cfg.LLM_DIR.exists() or not any(cfg.LLM_DIR.glob("*.jsonl")):
        print("WARNING: E8 skipped - no LLM cache in data/llm/ (LLM_* variables were not set)")
        return
    raise NotImplementedError("E8 analysis is not implemented: no LLM cache was available")


def run_runtime(args):
    """E7: median fitting time (sequential, single-threaded BLAS) and log-log slope."""
    import synthetic
    out = out_dir(args)
    groups = [10, 20] if args.quick else cfg.RUNTIME_GROUPS
    reps = 1 if args.quick else cfg.RUNTIME_REPEATS
    methods = cfg.METHODS + cfg.NEW_METHODS
    rows = [r for G in groups for r in synthetic.runtime_task((G, methods, reps))]
    rt = pd.DataFrame(rows)
    write_csv(rt, out / "runtime.csv")
    fit = []
    for method, sub in rt.groupby("method", sort=False):
        k, b = np.polyfit(np.log(sub.N), np.log(sub.seconds), 1)
        fit.append(dict(method=method, slope_loglog=k, intercept=b))
    write_csv(pd.DataFrame(fit), out / "runtime_fit.csv")


METHOD_RU = {"bt": "BT без признаков", "features": "только признаки",
             "hierarchical": "иерархическая BT", "bt_fixed": r"BT, $\lambda = 1/V$",
             "hier_oracle": r"оракул (известное $\mu$)",
             "hier_llm": "иерархическая BT + LLM", "hier_all": "иерархическая BT + все"}
TABLE_METRICS = [("tau", r"$\tau$", "max"), ("top1", "top-1", "max"), ("nll_ctrl", "NLL (контроль)", "min")]


def make_tables(args):
    """E6: tables/model_comparison.{csv,tex} from the saved results (no recomputation)."""
    res = out_dir(args)
    tdir = cfg.TABLES_DIR / "quick" if args.quick else cfg.TABLES_DIR
    rows = []
    e1 = res / "e1_raw.csv"
    if e1.exists():
        raw = pd.read_csv(e1)
        m = cfg.TABLE_SYNTH_BUDGET
        for rho in cfg.TABLE_SYNTH_RHOS:
            sub = raw[(raw.rho == rho) & (raw.budget == m)]
            for method in cfg.METHODS + cfg.NEW_METHODS:
                for metric, _, _ in TABLE_METRICS:
                    v = sub[(sub.method == method) & (sub.metric == metric)].value
                    if len(v):
                        rows.append(dict(slice=f"синтетика, ρ = {rho:g}, m = {m:g}", method=method,
                                         metric=metric, mean=v.mean(), sd=v.std(ddof=1), n=len(v)))
    real = res / "real_raw.csv"
    if real.exists():
        raw = pd.read_csv(real)
        if "feature_set" in raw:
            raw = raw[raw.feature_set.isin(["all", "none"])]
        for m in cfg.TABLE_REAL_BUDGETS:
            sub = raw[raw.budget == m]
            for method in sub.method.unique():
                for metric, _, _ in TABLE_METRICS:
                    v = sub[(sub.method == method) & (sub.metric == metric)].value
                    if len(v):
                        rows.append(dict(slice=f"реальные данные, m = {m:g}", method=method,
                                         metric=metric, mean=v.mean(), sd=v.std(ddof=1), n=len(v)))
    if not rows:
        return
    tab = pd.DataFrame(rows)
    write_csv(tab, tdir / "model_comparison.csv")
    order = list(METHOD_RU)
    lines = [r"\begin{tabular}{l" + "c" * len(TABLE_METRICS) + "}", r"\toprule",
             "Метод & " + " & ".join(h for _, h, _ in TABLE_METRICS) + r" \\", r"\midrule"]
    for k, (sl, sub) in enumerate(tab.groupby("slice", sort=False)):
        if k:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(TABLE_METRICS) + 1,
                                                                 sl.replace("ρ", r"$\rho$")))
        best = {}
        for metric, _, how in TABLE_METRICS:
            v = sub[sub.metric == metric].set_index("method")["mean"].round(3)
            best[metric] = v.max() if how == "max" else v.min()
        for method in sorted(sub.method.unique(), key=order.index):
            cells = []
            for metric, _, _ in TABLE_METRICS:
                r = sub[(sub.method == method) & (sub.metric == metric)].iloc[0]
                txt = f"{r['mean']:.3f} " + r"$\pm$" + f" {r['sd']:.3f}"
                if round(r["mean"], 3) == best[metric]:
                    txt = r"\textbf{" + f"{r['mean']:.3f}" + r"} $\pm$ " + f"{r['sd']:.3f}"
                cells.append(txt)
            lines.append(METHOD_RU[method] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (tdir / "model_comparison.tex").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def design_mse_vs_risk(args):
    """Derived from saved E3 results (no recomputation): measured contrast MSE of bt_fixed and
    hier_oracle divided by the 'full' expected risk at the same m (log-log interpolation on MC_GRID)."""
    res = out_dir(args)
    raw_p, risk_p = res / "designs_raw.csv", res / "designs_mc_risk.csv"
    if not (raw_p.exists() and risk_p.exists()):
        return
    raw = pd.read_csv(raw_p)
    risk = pd.read_csv(risk_p)
    raw = raw[(raw.metric == "mse") & raw.method.isin(["bt_fixed", "hier_oracle"])]
    mean = raw.groupby(["design", "rho", "method", "budget"]).value.mean().reset_index()
    rows = []
    for r in mean.itertuples(index=False):
        prior = "bt" if r.method == "bt_fixed" else f"h{r.rho:g}"
        rk = risk[(risk.design == r.design) & (risk["mode"] == "full") & (risk.prior == prior)]
        rk = rk.sort_values("m")
        pred = float(np.exp(np.interp(np.log(r.budget), np.log(rk.m), np.log(rk.risk))))
        rows.append(dict(design=r.design, rho=r.rho, method=r.method, budget=r.budget,
                         mse_measured=r.value, risk_full=pred, ratio=r.value / pred))
    write_csv(pd.DataFrame(rows), res / "designs_mse_vs_risk.csv")
