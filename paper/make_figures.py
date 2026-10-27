"""Result figures for the paper, computed from results/*.jsonl.

Usage (from the repository root):
  python paper/make_figures.py        -> paper/figures/success_by_vis.pdf, paper/figures/calibration.pdf
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "paper", "figures")
VIS = (10.0, 15.0, 20.0, 25.0)
RULES = [("voi", "Ours", "#2a78d6", ""), ("unc", "Max. shape uncertainty", "#eb6834", "//"), ("rand", "Random", "#1baf7a", "..")]
INK, MUTED, GRID = "#1d2120", "#5f6763", "#e2e3dd"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "hatch.linewidth": 0.6, "pdf.fonttype": 42})


def load(pattern):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, pattern))) for l in open(f)]


def success_by_vis():
    by = {}
    for r in load("results/tsel_*.jsonl"):
        by.setdefault(r["i"], {})[r["rule"]] = r
    ids = sorted(i for i, v in by.items() if len(v) == 3)
    vis = np.array([by[i]["voi"]["vis"] for i in ids])
    groups = [f"{v:.0f} mm" for v in VIS] + ["All"]
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    marks = {"voi": "o", "unc": "s", "rand": "^"}
    w = 0.22
    for j, (key, name, col, _) in enumerate(RULES):         # dot plot: the axis need not start at zero
        s = np.array([by[i][key]["steps"][0]["success"] for i in ids], float)
        vals = [s[vis == v].mean() * 100 for v in VIS] + [s.mean() * 100]
        x = np.arange(len(groups)) + (j - 1) * w
        ax.plot(x, vals, linestyle="none", marker=marks[key], markersize=6, color=col, markeredgecolor="white",
                markeredgewidth=0.8, label=name, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v - 2.2, f"{v:.0f}", ha="center", va="top", fontsize=5.5, color=INK)
    ax.axvline(len(VIS) - 0.5, color=GRID, linewidth=0.8)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_xlabel("Visible length of the flange", color=MUTED)
    ax.set_ylabel("Success after 1 touch (%)")
    ax.set_ylim(70, 101); ax.set_yticks([70, 80, 90, 100])
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "success_by_vis.pdf"))
    print("success_by_vis.pdf", len(ids), "layouts")


def calibration():
    R = load("results/branch_[0-9].jsonl")
    P = np.array([s["p"] for r in R for s in r["steps"]]); S = np.array([s["success"] for r in R for s in r["steps"]], float)
    edges = [0, .6, .7, .8, .9, .95, 1.0001]
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle="--", zorder=1, label="Perfect calibration")
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (P >= lo) & (P < hi)
        if m.sum():
            xs.append(P[m].mean()); ys.append(S[m].mean()); ns.append(int(m.sum()))
    ax.plot(xs, ys, color="#2a78d6", linewidth=2, marker="o", markersize=5, zorder=3, label="Ours (600 grasps)")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(-4, 6), ha="right", fontsize=6, color=MUTED)
    brier = float(np.mean((P - S) ** 2))
    ax.set_xlabel("Predicted success probability (particle average)")
    ax.set_ylabel("Observed success rate")
    ax.set_xlim(0.2, 1.02); ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.legend(loc="upper left", fontsize=6.5, frameon=False)
    ax.text(0.98, 0.04, f"Brier score {brier:.3f}", transform=ax.transAxes, ha="right", fontsize=7, color=INK)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "calibration.pdf"))
    print("calibration.pdf", len(P), "grasps, Brier", round(brier, 3), list(zip(ns, np.round(xs, 2), np.round(ys, 2))))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    success_by_vis()
    calibration()
