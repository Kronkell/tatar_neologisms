"""LaTeX tables for the paper, built only from saved results (run main.py first).

    python code/paper_tables.py     # -> tables/theory_check.tex, tables/designs.tex

Like plots.py, this never re-runs experiments.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "tables"


def f2(x):
    return f"{x:.2f}".replace(".", "{,}")


def ci(v, lo, hi):
    return f"{f2(v)} \\scriptsize[{f2(lo)}; {f2(hi)}]"


def theory_table():
    d = pd.read_csv(RES / "theory_check.csv").sort_values(["rho", "m0"])
    rows = []
    for _, r in d.iterrows():
        rows.append(" & ".join([
            f"{r.rho:g}".replace(".", "{,}"), f"{int(r.m0)}",
            f2(r.mc_mean_field), f2(r.mc_design), f2(r.mc_full),
            ci(r.meas_oracle_mse, r.meas_oracle_mse_lo, r.meas_oracle_mse_hi),
            f"${(r.meas_oracle_mse / r.mc_full - 1) * 100:+.0f}$\\%",
            ci(r.meas_hier_tau, r.meas_hier_tau_lo, r.meas_hier_tau_hi),
        ]) + r" \\")
    body = "\n".join(rows)
    return (r"""\begin{tabular}{cc ccc cc c}
\toprule
& & \multicolumn{3}{c}{прогноз до эксперимента} & \multicolumn{2}{c}{оракулы, MSE} & иерарх.\ BT, $\tau$ \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-8}
$\rho$ & $m_0$ & ср.\ поле \eqref{eq:closed} & + план & + инф.\ пар & измерение & откл. & измерение \\
\midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n")


def designs_table():
    d = pd.read_csv(RES / "designs_savings.csv")
    d = d[d.m0 == 1].copy()
    order = {"random": 0, "cycle": 1, "star": 2}
    names = {"random": "случайные пары", "cycle": "цикл", "star": "звезда"}
    d["o"] = d.design.map(order)
    d = d.sort_values(["o", "rho"])
    rows = []
    for _, r in d.iterrows():
        bounds = f"[{f2(r.bound_lo)}; {f2(r.bound_hi)}]"
        rows.append(" & ".join([
            names[r.design], f"{r.rho:g}".replace(".", "{,}"), bounds, f2(r.mc_mean_field), f2(r.mc_full),
            ci(r.meas_oracle_mse, r.meas_oracle_mse_lo, r.meas_oracle_mse_hi),
            ci(r.meas_hier_tau, r.meas_hier_tau_lo, r.meas_hier_tau_hi),
        ]) + r" \\")
    body = "\n".join(rows)
    return (r"""\begin{tabular}{lc c cc cc}
\toprule
план & $\rho$ & границы \eqref{eq:bounds} & ср.\ поле & full & оракулы, MSE & иерарх.\ BT, $\tau$ \\
\midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    TAB.mkdir(exist_ok=True)
    (TAB / "theory_check.tex").write_text(theory_table(), encoding="utf-8", newline="\n")
    (TAB / "designs.tex").write_text(designs_table(), encoding="utf-8", newline="\n")
    print("saved", TAB / "theory_check.tex", TAB / "designs.tex")


if __name__ == "__main__":
    main()
