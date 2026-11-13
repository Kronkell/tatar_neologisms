"""Figures from saved results (run main.py first).

    python code/plots.py            # figures/main_figure{,_slide}.{pdf,png}
    python code/plots.py --quick    # from results/quick/

Only reads results/*.csv - never re-runs the experiment.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import config as cfg

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
STYLE = {  # fixed categorical order: slot 1 blue, slot 2 orange, slot 3 aqua
    "hierarchical": dict(color="#2a78d6", marker="o", label="иерархическая BT (предлагаемая)"),
    "bt": dict(color="#eb6834", marker="s", label="BT без признаков"),
    "features": dict(color="#1baf7a", marker="^", label="только признаки"),
}
SAVING_SHADES = {1: "#1c5cab", 2: "#3987e5", 4: "#86b6ef"}  # ordinal blue ramp

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.5,
    "lines.markersize": 4, "legend.frameon": False, "savefig.dpi": 300,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})


def budget_axis(ax, budgets):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(budgets))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def panel_curves(ax, curves, savings, rho, compact=False):
    df = curves[curves.rho == rho]
    budgets = sorted(df.budget.unique())
    for method in ["hierarchical", "bt", "features"]:
        d = df[df.method == method].sort_values("budget")
        st = STYLE[method]
        ax.fill_between(d.budget, d.tau_lo, d.tau_hi, color=st["color"], alpha=0.15, lw=0)
        label = st["label"].replace("(предлагаемая)", "(наша)") if compact else st["label"]
        ax.plot(d.budget, d.tau_mean, color=st["color"], marker=st["marker"],
                label=label, zorder=3,
                markeredgecolor="white", markeredgewidth=0.5)

    # arrow: the same quality with m0 = 1 (hierarchical) vs m_BT (plain BT)
    m0 = 1
    s = savings[(savings.rho == rho) & (savings.m0 == m0)].iloc[0]
    tau0 = df[(df.method == "hierarchical") & (df.budget == m0)].tau_mean.item()
    m_bt = m0 * s.saving
    ax.annotate("", xy=(m_bt, tau0), xytext=(m0, tau0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=3, shrinkB=1,
                                mutation_scale=7), zorder=5)
    text = (f"в {s.saving:.1f} раза\nменьше сравнений" if compact
            else f"то же качество —\nв {s.saving:.1f} раза меньше\nсравнений")
    ax.annotate(text,
                xy=(np.sqrt(m0 * m_bt), tau0 + 0.005), xytext=(0.47, 0.84 if compact else 0.70),
                ha="left", va="center", fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6,
                                connectionstyle="arc3,rad=-0.2", shrinkA=2, shrinkB=2))

    budget_axis(ax, budgets)
    ax.set_ylim(0.15, 0.92)
    ax.set_xlabel(r"сравнений на кандидата, $m$")
    ax.set_ylabel(r"качество, $\tau$" if compact else r"качество ранжирования, $\tau$")
    ax.set_title(rf"а) качество от бюджета, $\rho$ = {rho:g}" if compact
                 else rf"а) качество от бюджета сравнений, $\rho$ = {rho:g}", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(loc="lower right", fontsize=7.2, handlelength=1.6 if compact else 2.0,
              borderaxespad=0.2)


def panel_savings(ax, savings, main_rho, compact=False):
    for m0, d in savings.groupby("m0"):
        d = d.sort_values("rho")
        c = SAVING_SHADES.get(m0, "#2a78d6")
        ax.errorbar(d.rho, d.saving, yerr=[d.saving - d.saving_lo, d.saving_hi - d.saving],
                    color=c, marker="o", capsize=1.8, lw=1.5, elinewidth=0.7,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        last = d.iloc[-1]
        ax.text(last.rho + 0.03, last.saving, rf"$m_0$ = {m0:g}", va="center",
                fontsize=7.5, color=INK)
    hl = savings[(savings.rho == main_rho) & (savings.m0 == 1)]
    if len(hl):
        ax.plot(hl.rho, hl.saving, "o", ms=9, mfc="none", mec=INK, mew=0.9, zorder=4)
        ax.annotate("стрелка на (а)", xy=(hl.rho.item() + 0.02, hl.saving.item() * 0.93),
                    xytext=(0.57, 1.2 if compact else 1.45), fontsize=7, color=INK_2, ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6, shrinkA=1, shrinkB=3))
    ax.axhline(1.0, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    ax.text(0.99, 0.95, "нет выигрыша", fontsize=7, color=INK_2, va="top", ha="right")
    ax.set_yscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.set_ylim(0.6, 16)
    ax.set_xlim(0.05, 1.04)
    ax.set_xticks(sorted(savings.rho.unique()))
    if compact:
        ax.set_xlabel(r"$\rho$ — доля, объяснимая признаками")
        ax.set_ylabel(r"экономия, $m_{\mathrm{BT}}\,/\,m_0$")
        ax.set_title("б) экономия сравнений", loc="left")
    else:
        ax.set_xlabel(r"доля разброса предпочтений, объяснимая признаками, $\rho$")
        ax.set_ylabel(r"экономия сравнений, $m_{\mathrm{BT}}\,/\,m_0$")
        ax.set_title("б) во сколько раз признаки сокращают опрос", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="FIG",
                        help="draw only these figures (main_figure, fig_theory_check, fig_designs, "
                             "fig_real, fig_real_theory, fig_complexity)")
    args = parser.parse_args()
    res_dir = cfg.RESULTS_DIR / "quick" if args.quick else cfg.RESULTS_DIR
    draw_main = not args.only or "main_figure" in args.only
    if draw_main:
        curves = pd.read_csv(res_dir / "curves.csv")
        savings = pd.read_csv(res_dir / "savings.csv")
        summary = json.loads((res_dir / "summary.json").read_text())
        main_rho = summary["main_rho"] if summary["main_rho"] in set(curves.rho) else curves.rho.min()

    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    quick = "_quick" if args.quick else ""
    # main_figure: \textwidth of the paper (about 6.3 in);
    # main_figure_slide: right-hand 70% of the one-slide talk, shorter labels
    for name, figsize, compact in [("main_figure", (6.4, 2.45), False),
                                   ("main_figure_slide", (5.0, 2.2), True)]:
        if not draw_main:
            break
        fig, (a, b) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.1, 1]))
        panel_curves(a, curves, savings, main_rho, compact)
        panel_savings(b, savings, main_rho, compact)
        fig.tight_layout(w_pad=1.2 if compact else 1.6, pad=0.3)
        for ext in ["pdf", "png"]:
            fig.savefig(cfg.FIGURES_DIR / f"{name}{quick}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print("saved", cfg.FIGURES_DIR / f"{name}{quick}.pdf")
    extra_figures(res_dir, quick, args.only)


# =============================================================================
# Figures added after the first report (read results/ only)
# =============================================================================
DESIGN_STYLE = {"random": dict(color="#3b3b3b", marker="o", label="случайные пары"),
                "cycle": dict(color="#c2185b", marker="s", label="цикл"),
                "star": dict(color="#7b61c9", marker="^", label="звезда")}
MODE_STYLE = {"mean_field": dict(ls=":", label="теория: среднее поле"),
              "design": dict(ls="--", label="+ случайность плана"),
              "full": dict(ls="-", label="+ неравная информация пар")}


def _save(fig, name, quick):
    fig.tight_layout(pad=0.3)
    for ext in ["pdf", "png"]:
        fig.savefig(cfg.FIGURES_DIR / f"{name}{quick}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("saved", cfg.FIGURES_DIR / f"{name}{quick}.pdf")


def _logy_saving(ax, ticks=(1, 2, 5, 10, 20, 50)):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax.grid(axis="y", color=GRID, lw=0.5)


def _err(sub, col):
    return [sub[col] - sub[col + "_lo"], sub[col + "_hi"] - sub[col]]


def fig_theory_check(res_dir, quick):
    tc = pd.read_csv(res_dir / "theory_check.csv")
    path = res_dir / "designs_savings.csv"
    ds = pd.read_csv(path) if path.exists() else tc.assign(design="random")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    d = tc[tc.m0 == 1].sort_values("rho")
    for mode, st in MODE_STYLE.items():
        a.plot(d.rho, d[f"mc_{mode}"], color=INK_2, ls=st["ls"], lw=1.1, label=st["label"])
    a.errorbar(d.rho, d.meas_oracle_mse, yerr=_err(d, "meas_oracle_mse"),
               color=STYLE["hierarchical"]["color"], marker="o", ls="none", capsize=1.8,
               elinewidth=0.7, markeredgecolor="white", markeredgewidth=0.5, zorder=3,
               label="измерение (оракулы, MSE)")
    _logy_saving(a)
    a.set_xticks(sorted(d.rho.unique()))
    a.set_xlabel(r"доля, объяснимая признаками, $\rho$")
    a.set_ylabel(r"экономия, $m_{\mathrm{BT}}/m_0$")
    a.set_title(r"а) оракулы, MSE, $m_0$ = 1", loc="left")
    a.legend(fontsize=6.8, loc="upper left")

    lim = [0.9, 1.15 * max(ds.mc_full.max(), ds.meas_oracle_mse.max(), ds.meas_hier_tau.max())]
    b.plot(lim, lim, color=INK_2, lw=0.8, ls=(0, (4, 3)))
    for design in ["random", "cycle", "star"]:
        sub = ds[ds.design == design]
        if not len(sub):
            continue
        st = DESIGN_STYLE[design]
        b.scatter(sub.mc_full, sub.meas_oracle_mse, color=st["color"], marker=st["marker"], s=16,
                  label=st["label"] + ": оракулы, MSE", zorder=3, edgecolors="white",
                  linewidths=0.4)
        b.scatter(sub.mc_full, sub.meas_hier_tau, facecolors="none", edgecolors=st["color"],
                  marker=st["marker"], s=16, linewidths=0.8,
                  label=st["label"] + r": иерарх. BT, $\tau$")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(lim)
    b.set_ylim(lim)
    for axis in (b.xaxis, b.yaxis):
        axis.set_major_locator(FixedLocator([1, 2, 5, 10, 20]))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_xlabel("предсказание (режим full)")
    b.set_ylabel("измерение")
    b.set_title(r"б) все $\rho$, $m_0$ и планы", loc="left")
    b.legend(fontsize=5.8, loc="upper left", handletextpad=0.3)
    b.grid(color=GRID, lw=0.5)
    _save(fig, "fig_theory_check", quick)


def fig_designs(res_dir, quick):
    ds = pd.read_csv(res_dir / "designs_savings.csv")
    ds = ds[ds.m0 == 1]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, (key, title) in zip(axes, [("oracle_mse", "а) оракулы, MSE"),
                                       ("hier_tau", r"б) иерархическая BT против BT, $\tau$")]):
        for k, design in enumerate(["random", "cycle", "star"]):
            sub = ds[ds.design == design].sort_values("rho")
            if not len(sub):
                continue
            st = DESIGN_STYLE[design]
            ax.fill_between(sub.rho, sub.bound_lo, np.minimum(sub.bound_hi, 100),
                            color=st["color"], alpha=0.08, lw=0)
            ax.plot(sub.rho, sub.mc_full, color=st["color"], lw=1.2)
            ax.plot(sub.rho, sub.mc_mean_field, color=st["color"], lw=0.9, ls=":")
            ax.errorbar(sub.rho + (k - 1) * 0.012, sub[f"meas_{key}"], yerr=_err(sub, f"meas_{key}"),
                        color=st["color"], marker=st["marker"], ls="none", capsize=1.5,
                        elinewidth=0.7, ms=3.5, label=st["label"])
        _logy_saving(ax, ticks=(1, 2, 3, 5, 10, 20))
        ax.set_ylim(1.0, 25)
        ax.set_xticks(sorted(ds.rho.unique()))
        ax.set_xlabel(r"$\rho$")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel(r"экономия при $m_0$ = 1")
    handles = [plt.Line2D([], [], color=INK_2, lw=1.2, label="теория (по MSE), full"),
               plt.Line2D([], [], color=INK_2, lw=0.9, ls=":", label="теория (по MSE), среднее поле"),
               plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.12, label="спектральные границы")]
    axes[0].legend(fontsize=6.5, loc="upper left")
    axes[1].legend(handles=handles, fontsize=6.5, loc="upper left")
    _save(fig, "fig_designs", quick)


def fig_real(res_dir, quick):
    cur = pd.read_csv(res_dir / "real_curves.csv")
    if "feature_set" in cur:
        cur = cur[cur.feature_set.isin(["all", "none"])]
    meta = json.loads((res_dir / "real_dataset.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.5))
    budgets = sorted(cur.budget.unique())
    for method in ["hierarchical", "bt", "features"]:
        st = STYLE[method]
        d = cur[(cur.method == method) & (cur.metric == "tau")].sort_values("budget")
        a.fill_between(d.budget, d.lo, d.hi, color=st["color"], alpha=0.15, lw=0)
        a.plot(d.budget, d["mean"], color=st["color"], marker=st["marker"], label=st["label"],
               markeredgecolor="white", markeredgewidth=0.5)
        for metric, ls in (("nll_train", "--"), ("nll_ctrl", "-")):
            d = cur[(cur.method == method) & (cur.metric == metric)].sort_values("budget")
            b.plot(d.budget, d["mean"], color=st["color"], marker=st["marker"], ls=ls, ms=3,
                   markeredgecolor="white", markeredgewidth=0.4)
    c = meta["agreement_A_B_tau"]["mean"]
    a.axhline(c, color=INK_2, lw=0.9, ls=(0, (4, 3)))
    a.text(budgets[0], c + 0.01, "согласие групп A и B", fontsize=7, color=INK_2, va="bottom")
    for ax in (a, b):
        budget_axis(ax, budgets)
        ax.set_xlabel(r"сравнений на слово, $m$")
        ax.grid(axis="y", color=GRID, lw=0.5)
    a.set_ylabel(r"$\tau$ с оценками аннотаторов B")
    a.set_title("а) качество ранжирования", loc="left")
    a.legend(fontsize=6.8, loc="lower right")
    b.set_ylabel("NLL на сравнение")
    b.set_title("б) кривые обучения", loc="left")
    b.legend(handles=[plt.Line2D([], [], color=INK_2, ls="--", label="обучение (аннотаторы A)"),
                      plt.Line2D([], [], color=INK_2, ls="-", label="контроль (аннотаторы B)")],
             fontsize=6.8, loc="center right", bbox_to_anchor=(1.0, 0.3))
    _save(fig, "fig_real", quick)


def fig_complexity(res_dir, quick):
    comp = pd.read_csv(res_dir / "real_complexity.csv")
    path = pd.read_csv(res_dir / "real_lambda_path.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for method in ["hierarchical", "features"]:
        st = STYLE[method]
        for m, ls in zip(sorted(comp.budget.unique()), ["--", "-"]):
            d = comp[(comp.method == method) & (comp.metric == "tau") & (comp.budget == m)]
            g = d.groupby("d").value.agg(["mean", "std"]).reset_index()
            a.errorbar(g.d, g["mean"], yerr=g["std"], color=st["color"], marker=st["marker"],
                       ls=ls, capsize=1.5, elinewidth=0.6, ms=3.5,
                       label=f"{st['label'].replace(' (предлагаемая)', '')}, m = {m:g}")
    a.set_xticks(sorted(comp.d.unique()))
    a.set_xlabel("число признаков d (zipf, длина, слоги, charLM)")
    a.set_ylabel(r"$\tau$ с оценками аннотаторов B")
    a.set_title("а) сложность модели", loc="left")
    a.legend(fontsize=6, loc="lower right")
    a.grid(axis="y", color=GRID, lw=0.5)
    g = path.groupby("lam")[["nll_train", "nll_ctrl", "nll_holdout_annotator",
                             "nll_holdout_comparisons"]].mean().reset_index()
    c = STYLE["hierarchical"]["color"]
    b.plot(g.lam, g.nll_train, color=c, ls="--", marker="o", ms=3, label="обучение (пул A)")
    b.plot(g.lam, g.nll_ctrl, color=c, ls="-", marker="o", ms=3, label="контроль (аннотаторы B)")
    b.plot(g.lam, g.nll_holdout_annotator, color=INK_2, ls="-.", marker="s", ms=2.5,
           label="отложенный аннотатор A (выбор)")
    b.plot(g.lam, g.nll_holdout_comparisons, color=INK_2, ls=":", marker=".", ms=3,
           label="отложенные 20 % сравнений (прежний выбор)")
    for col, ls, txt in (("selected_by_annotator", "-.", "по аннотатору"),
                         ("selected_by_comparisons", ":", "по сравнениям")):
        sel = path[path[col]].lam.mode().iloc[0]
        b.axvline(sel, color=INK_2, lw=0.7, ls=ls)
        b.annotate(txt, xy=(sel, 0.03), xycoords=("data", "axes fraction"), fontsize=6.3,
                   color=INK_2, va="bottom", ha="right", rotation=90)
    b.set_xscale("log")
    b.set_xlabel(r"$\lambda$ (иерархическая BT, $m$ = 2)")
    b.set_ylabel("NLL на сравнение")
    b.set_title("б) регуляризация", loc="left")
    b.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2,
             columnspacing=0.8, handlelength=2.2)
    b.grid(axis="y", color=GRID, lw=0.5)
    _save(fig, "fig_complexity", quick)


SET_STYLE = {"charlm": dict(color="#8c6bb1", label="charLM"),
             "syllables": dict(color="#e7298a", label="слоги"),
             "length": dict(color="#66a61e", label="длина"),
             "zipf": dict(color="#d4a106", label="zipf"),
             "zipf+length": dict(color="#1baf7a", label="zipf + длина (≈ все 4)"),
             "all": dict(color="#2a78d6", label="все 4 признака")}


def fig_real_theory(res_dir, quick):
    """Predicted (full and mean field) vs measured saving by tau on real data."""
    sv = pd.read_csv(res_dir / "real_savings.csv")
    sv = sv[(sv.target == "hierarchical") & (sv.metric == "tau")]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    pts = []
    for fs, st in SET_STYLE.items():
        d = sv[sv.feature_set == fs]
        for col, filled in (("pred_full_seed0", True), ("pred_mean_field", False)):
            for r in d.itertuples(index=False):
                x = getattr(r, col)
                if not np.isfinite(x) or not np.isfinite(r.saving):
                    continue
                pts.append((x, r.saving))
                kw = dict(color=st["color"]) if filled else dict(facecolors="none",
                                                                  edgecolors=st["color"])
                big = fs == "zipf+length"   # coincides with "all": drawn as a larger ring below
                ax.scatter([x], [r.saving], marker="o", s=55 if big else 18, linewidths=0.9,
                           zorder=2 if big else 3, **kw)
                if r.censored:
                    ax.annotate("", xy=(x, r.saving * 1.6), xytext=(x, r.saving),
                                arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=0.8,
                                                mutation_scale=6, shrinkA=2, shrinkB=0))
    if pts:
        v = np.array(pts)
        lim = [0.8 * v.min(), 2.2 * v.max()]
        ax.plot(lim, lim, color=INK_2, lw=0.8, ls=(0, (4, 3)))
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"предсказанная экономия $m_{\mathrm{BT}}/m_0$")
    ax.set_ylabel(r"измеренная экономия по $\tau$")
    ax.set_title(r"реальные данные: иерархическая BT против BT, $m_0$ = 0.25…4", loc="left")
    ax.grid(color=GRID, lw=0.5)
    handles = [plt.Line2D([], [], color=st["color"], marker="o", ls="none", label=st["label"])
               for st in SET_STYLE.values()]
    handles += [plt.Line2D([], [], color=INK_2, marker="o", ls="none",
                           label="теория full (разбиение 0)"),
                plt.Line2D([], [], color=INK_2, marker="o", mfc="none", ls="none",
                           label="теория, среднее поле\n(10 разбиений)"),
                plt.Line2D([], [], color=INK_2, marker=r"$\uparrow$", ls="none", ms=7,
                           label="цензурировано:\nнижняя граница")]
    ax.legend(handles=handles, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, "fig_real_theory", quick)


def extra_figures(res_dir, quick, only=None):
    for fn, need in [(fig_theory_check, "theory_check.csv"), (fig_designs, "designs_savings.csv"),
                     (fig_real, "real_curves.csv"), (fig_real_theory, "real_savings.csv"),
                     (fig_complexity, "real_complexity.csv")]:
        if only and fn.__name__ not in only:
            continue
        if (res_dir / need).exists():
            fn(res_dir, quick)

if __name__ == "__main__":
    main()
