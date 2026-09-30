import json, glob
import numpy as np
from scipy.stats import binomtest
R = [json.loads(l) for f in glob.glob("results/tsel_*.jsonl") for l in open(f)]
by = {}
for r in R:
    by.setdefault(r["i"], {})[r["rule"]] = r
ids = sorted(i for i, v in by.items() if len(v) == 3)
print("配置", len(ids))
rng = np.random.default_rng(0)
VIS = (10.0, 15.0, 20.0, 25.0)
for k in (1, 2):
    print(f"--- {k}回触った時点")
    S = {ru: np.array([by[i][ru]["steps"][k - 1]["success"] for i in ids], float) for ru in ("voi", "unc", "rand")}
    SD = {ru: np.array([by[i][ru]["steps"][k - 1]["body_front_sd_mm"] for i in ids]) for ru in S}
    vis = np.array([by[i]["voi"]["vis"] for i in ids])
    for ru in S:
        per = " ".join(f"{S[ru][vis == v].mean():.2f}" for v in VIS)
        print(f"  {ru:4s} 成功率 {S[ru].mean():.3f}  見え方別 {per}  本体の手前の縁のばらつき(中央値) {np.median(SD[ru]):.1f}mm")
    for other in ("unc", "rand"):
        dlt = S["voi"] - S[other]
        bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(3000)]
        w, l = int((dlt > 0).sum()), int((dlt < 0).sum())
        p = binomtest(w, w + l, 0.5).pvalue if w + l else 1.0       # McNemar の正確検定
        print(f"  voi − {other}: {dlt.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  voi勝ち{w} 負け{l}  McNemar p={p:.3f}")
    for v in VIS:
        m = vis == v; d = S["voi"][m] - S["unc"][m]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
        print(f"  見え方{v:.0f}mm voi − unc: {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
