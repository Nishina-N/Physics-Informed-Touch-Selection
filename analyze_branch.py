"""分岐評価の集計：A（0回）・Bn（n回）・C(τ)（成功確率が初めてτ以上になった時点、上限4回）。"""
import json, glob, sys
import numpy as np

pat = sys.argv[1] if len(sys.argv) > 1 else "results/branch_[0-9].jsonl"
R = [json.loads(l) for f in glob.glob(pat) for l in open(f)]
R.sort(key=lambda r: r["i"])
VIS = (10.0, 15.0, 20.0, 25.0)


def pick(r, rule):
    st = r["steps"]
    if rule[0] == "B":
        return st[rule[1]]
    for s in st:                                   # C
        if s["p"] >= rule[1] or s["k"] == len(st) - 1:
            return s


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


rules = [("A", ("B", 0)), ("B1", ("B", 1)), ("B2", ("B", 2)), ("B3", ("B", 3)), ("B4", ("B", 4))] + \
        [(f"C{t}", ("C", t)) for t in (0.7, 0.8, 0.85, 0.9, 0.95)]
print(f"配置 {len(R)}（見え方ごと {[sum(1 for r in R if r['vis'] == v) for v in VIS]}）")
print("条件 | 成功率 [95%区間] | 触る回数 | 見え方別 成功率/回数")
for name, rule in rules:
    ch = [pick(r, rule) for r in R]
    s = [c["success"] for c in ch]; k = [c["k"] for c in ch]
    ci = boot_ci(s)
    per = []
    for v in VIS:
        idx = [j for j, r in enumerate(R) if r["vis"] == v]
        per.append(f"{np.mean([s[j] for j in idx]):.2f}/{np.mean([k[j] for j in idx]):.1f}")
    print(f"{name:5s} | {np.mean(s):.3f} [{ci[0]:.2f},{ci[1]:.2f}] | {np.mean(k):.2f} | {' '.join(per)}")
# 較正：各ステップの予測 p と成否
P = np.array([s["p"] for r in R for s in r["steps"]]); S = np.array([s["success"] for r in R for s in r["steps"]])
print("較正（全ステップ）")
for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.01)]:
    m = (P >= lo) & (P < hi)
    if m.sum():
        print(f"  予測 {lo:.2f}〜{hi:.2f}: {m.sum()}件 予測平均 {P[m].mean():.2f} 実際 {S[m].mean():.2f}")
F = [s for r in R for s in r["steps"] if not s["success"]]
mv = [s["moved_mm"] for s in F if s["moved_mm"] is not None]
print(f"失敗した掴み {len(F)}件：部品の移動 中央値 {np.median(mv):.0f}mm、10mm以上 {np.mean(np.array(mv) > 10):.0%}")
tm = [s["touch_moved_mm"] for r in R for s in r["steps"] if "touch_moved_mm" in s]
print(f"触る {len(tm)}回：部品の移動 最大 {max(tm):.2f}mm")
