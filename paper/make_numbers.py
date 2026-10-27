"""Numbers of the learned-vs-geometric ablation, written as LaTeX macros (paper/numbers_geom.tex).

Compares Ours (results/tsel_*.jsonl, rule voi) with Ours-Geom (results/tselgeom_*.jsonl) on the same layouts.
Usage (from the repository root): python paper/make_numbers.py
"""
import glob, json, os
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.join(os.path.dirname(__file__), "..")
VIS = (10.0, 15.0, 20.0, 25.0)


def load(pat):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(ROOT, pat))) for l in open(f)]


def paired(a, b, rng):
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    w, l = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0
    return d.mean() * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100, w, l, p


def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def main():
    allr = load("results/tsel_*.jsonl")
    ours = {r["i"]: r for r in allr if r["rule"] == "voi"}
    unc = {r["i"]: r for r in allr if r["rule"] == "unc"}
    geom = {r["i"]: r for r in load("results/tselgeom_*.jsonl")}
    ids = sorted(set(ours) & set(geom))
    out = os.path.join(ROOT, "paper", "numbers_geom.tex")
    if len(ids) < 240:
        open(out, "w").write(
            "\\newcommand{\\GeomSummary}{[ablation running]}\n\\newcommand{\\GeomDiffShort}{[TBD]}\n"
            f"\\newcommand{{\\GeomResultsParagraph}}{{[Ablation running: {len(ids)} of 240 layouts.]}}\n")
        print("partial", len(ids)); return
    rng = np.random.default_rng(0)
    vis = np.array([ours[i]["vis"] for i in ids])
    lines = []
    res = {}
    for k in (1, 2):
        a = np.array([ours[i]["steps"][k - 1]["success"] for i in ids], float)
        g = np.array([geom[i]["steps"][k - 1]["success"] for i in ids], float)
        res[k] = (a, g, paired(a, g, rng))
    a, g, (m, lo, hi, w, l, p) = res[1]
    per = []
    for v in VIS:
        s = vis == v
        mm, ll, hh, *_ = paired(a[s], g[s], rng)
        per.append((v, a[s].mean() * 100, g[s].mean() * 100, mm, ll, hh))
    a2, g2, (m2, lo2, hi2, w2, l2, p2) = res[2]
    sd_o = np.median([ours[i]["steps"][0]["body_front_sd_mm"] for i in ids])
    sd_g = np.median([geom[i]["steps"][0]["body_front_sd_mm"] for i in ids])
    sign = lambda x: f"{float(f'{x / 100:.3f}') * 100:+.1f}".replace("-", "$-$")   # round like analyze_touchsel.py
    u = np.array([unc[i]["steps"][0]["success"] for i in ids], float)
    mu, lou, hiu, wu, lu, pu = paired(g, u, rng)
    br = lambda d: float(np.mean([(d[i]["steps"][0]["p"] - d[i]["steps"][0]["success"]) ** 2 for i in ids]))
    brier_o, brier_g = br(ours), br(geom)
    summ = (f"{g.mean()*100:.1f}\\%, only {sign(m)} points below ours (95\\% CI [{sign(lo)}, {sign(hi)}], ${fmt_p(p)}$)")
    para = (f"With the three-threshold model in place of $f_\\phi$ (\\Geom), success after one touch was {g.mean()*100:.1f}\\%, against "
            f"{a.mean()*100:.1f}\\% for \\Ours{{}} (paired difference {sign(m)} points, 95\\% CI [{sign(lo)}, {sign(hi)}]; discordant layouts "
            f"{w} vs.\\ {l}; McNemar ${fmt_p(p)}$). By visible length, \\Ours{{}} minus \\Geom{{}} was "
            + ", ".join(f"{sign(x[3])} [{sign(x[4])}, {sign(x[5])}] at {x[0]:.0f}\\,mm" for x in per)
            + f". After two touches, success was {a2.mean()*100:.1f}\\% versus {g2.mean()*100:.1f}\\% ({sign(m2)} [{sign(lo2)}, {sign(hi2)}], ${fmt_p(p2)}$). "
            f"\\Geom{{}} was still {sign(mu)} points above \\Unc{{}} ([{sign(lou)}, {sign(hiu)}]; {wu} vs.\\ {lu}; ${fmt_p(pu)}$). "
            f"The median spread of the body front edge after one touch was {sd_o:.1f}\\,mm for both, and the Brier score of the "
            f"predicted success at the executed grasp was {brier_o:.3f} (\\Ours) and {brier_g:.3f} (\\Geom).")
    open(out, "w").write(f"\\newcommand{{\\GeomSummary}}{{{summ}}}\n\\newcommand{{\\GeomDiffShort}}{{{sign(m)} points}}\n\\newcommand{{\\GeomVsUnc}}{{{sign(mu)} points}}\n\\newcommand{{\\GeomSucc}}{{{g.mean()*100:.1f}\\%}}\n"
                         f"\\newcommand{{\\GeomResultsParagraph}}{{{para}}}\n")
    print(open(out).read())


if __name__ == "__main__":
    main()


def rules_table():
    """Table of all touch rules (results/tsel, tselgeom, tselfixshape) and the fixed-shape macros."""
    by = {}
    for pat in ("results/tsel_*.jsonl", "results/tselgeom_*.jsonl", "results/tselfixshape_*.jsonl"):
        for r in load(pat):
            by.setdefault(r["i"], {})[r["rule"]] = r
    ids = sorted(i for i, v in by.items() if len(v) == 5)
    rng = np.random.default_rng(1)
    S = lambda ru, k: np.array([by[i][ru]["steps"][k]["success"] for i in ids], float)
    sign = lambda x: f"{float(f'{x / 100:.3f}') * 100:+.1f}".replace("-", "$-$")   # round like analyze_touchsel.py (3 decimals of a fraction)
    rows = [("voi", "\\Ours", "yes", "grasp (lookahead)", "network"),
            ("geom", "\\Geom", "yes", "grasp (lookahead)", "3 thresholds"),
            ("fixshape", "\\Fix", "no (nominal)", "grasp (lookahead)", "network"),
            ("unc", "\\Unc", "yes", "shape uncertainty", "network"),
            ("rand", "\\Rand", "yes", "random", "network")]
    # bootstrap streams chosen to reproduce the CIs printed by analyze_touchsel.py (unc, then rand on one stream)
    # and by main() above (geom); fixshape gets its own fresh stream
    rng_ur = np.random.default_rng(0)
    diffs = {"unc": paired(S("voi", 0), S("unc", 0), rng_ur)}
    diffs["rand"] = paired(S("voi", 0), S("rand", 0), rng_ur)
    diffs["geom"] = paired(S("voi", 0), S("geom", 0), np.random.default_rng(0))
    diffs["fixshape"] = paired(S("voi", 0), S("fixshape", 0), np.random.default_rng(0))
    body = []
    for key, name, shp, crit, model in rows:
        cells = [f"{S(key, k).mean() * 100:.1f}\\%" for k in (0, 1)]
        if key == "voi":
            diff = "---"
        else:
            m, lo, hi, w, l, p = diffs[key]
            diff = f"{sign(m)} [{sign(lo)}, {sign(hi)}]"
        body.append(f"{name} & {shp} & {crit} & {model} & {cells[0]} & {cells[1]} & {diff} \\\\")
    tab = ("\\begin{table*}[t]\n\\caption{Touch rules: what each estimates and how it chooses touches, and grasp success after one and two touches "
           f"({len(ids)} paired layouts). The last column is \\Ours{{}} minus the rule after one touch, with a 95\\% bootstrap CI.}}\n"
           "\\label{tab:rules}\n\\centering\n\\small\n\\begin{tabular}{@{}lllllll@{}}\n\\toprule\n"
           "Rule & Hidden shape estimated & Touch criterion & Success model & 1 touch & 2 touches & \\Ours{} $-$ rule \\\\\n\\midrule\n"
           + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    open(os.path.join(ROOT, "paper", "table_rules.tex"), "w").write(tab)
    vis = np.array([by[i]["voi"]["vis"] for i in ids])
    m, lo, hi, w, l, p = diffs["fixshape"]
    m10, lo10, hi10, *_ = paired(S("voi", 0)[vis == 10], S("fixshape", 0)[vis == 10], rng)
    mu, lou, hiu, wu, lu, pu = paired(S("fixshape", 0), S("unc", 0), rng)
    fx = S("fixshape", 0).mean() * 100
    mac = (f"\\newcommand{{\\FixSucc}}{{{fx:.1f}\\%}}\n"
           f"\\newcommand{{\\FixDiff}}{{{sign(m)} points (95\\% CI [{sign(lo)}, {sign(hi)}]; discordant {w} vs.\\ {l}; McNemar ${fmt_p(p)}$)}}\n"
           f"\\newcommand{{\\FixDiffShort}}{{{sign(m)} points}}\n"
           f"\\newcommand{{\\FixDiffTen}}{{{sign(m10)} [{sign(lo10)}, {sign(hi10)}]}}\n"
           f"\\newcommand{{\\FixVsUnc}}{{{sign(mu)} points ([{sign(lou)}, {sign(hiu)}]; ${fmt_p(pu)}$)}}\n")
    open(os.path.join(ROOT, "paper", "numbers_fix.tex"), "w").write(mac)
    print(tab); print(mac)


if __name__ == "__main__":
    rules_table()
