"""SmolVLM との組み合わせの比較：部品が2つ並ぶ場面で、選んだ部品を
  A：触らずに頭部カメラの推定だけで掴む（SmolVLM で選んでそのまま掴む）
  B：提案手法（voi）で1回触って確かめてから掴む（SmolVLM＋提案手法）
の把持成功率を比べる。どちらを選ぶかは SmolVLM の役目なので、ここでは2つの部品の両方を「選ばれた」として回す。
場面は vlm_pick.py と同じ（今の位置と右へ 10cm、隣の部品は動かない置物）。部品ごとの隠れたサイズ（フランジと本体の長さ）は
予測器の学習範囲から乱数で引き、見え方・向き・物理の値は run_branch.layout(i) を使う。
A は状態を複製して掴み（run_branch.branch_grasp）、元の状態から B を続けるので、同じ観測から両方を評価する。

使い方: python3 run_vlm_compare.py k n N → results/vlmcmp_k.jsonl（N 配置 × 部品2つ、n 並列の k 番目）
集計  : python3 run_vlm_compare.py --summary
"""
import sys, os, json, glob, gc
import numpy as np
import shelf as S
import perception as P
import partfilter as F
import build_scene as bs
import run_branch as RB
import vlm_pick as VP

COLORS = ["blue", "green"]


def sizes_for(i):
    rng = np.random.default_rng(7000 + i)
    return [(float(rng.uniform(*RB.FL_R)), float(rng.uniform(*RB.L_R))) for _ in VP.SLOTS]


def trial(i, slot):
    sizes = sizes_for(i)
    s, lay = VP.make_sim(i, slot, COLORS, sizes)
    rng = np.random.default_rng(5000 + i)
    F.use_predictor("predictor.pkl", s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    est, _ = P.init_particles_part(cam, cm, occl, RB.W, ztab, rng, (S.X_FRONT, S.Y0 + VP.SLOTS[slot]),
                                   RB.FL_R, RB.L_R, yaw0=yaw0)
    est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, RB.FL_R, rng)
    (gx, gy), p0 = F.best_grasp_m(est, F.axis_points(est))
    a = RB.branch_grasp(s, [gx, gy, ztab + RB.W / 2])                       # A：触らずに掴む
    gc_ = F.axis_points(est); tc = F.axis_points(est, step=0.003)
    tx, ty = F.choose_touch_m(est, gc_, tc)[0]                              # B：voi で1回触る
    r = s.touch([tx, ty])
    est = F.update(est, tx, ty, r["obs"], rng)
    (gx, gy), p1 = F.best_grasp_m(est, F.axis_points(est))
    b = RB.branch_grasp(s, [gx, gy, ztab + RB.W / 2])
    out = dict(i=i, slot=slot, vis=lay["vis"] * 1000, size_mm=[sizes[slot][0] * 1000, sizes[slot][1] * 1000],
               body_seen=seen, notouch=dict(success=a["success"], p=p0), touch=dict(success=b["success"], p=p1, obs=r["obs"]))
    del s; gc.collect()
    return out


def summary(pat="results/vlmcmp_*.jsonl"):
    R = [json.loads(l) for f in glob.glob(pat) for l in open(f)]
    if not R:
        print("結果がありません"); return
    A = np.array([r["notouch"]["success"] for r in R], float); B = np.array([r["touch"]["success"] for r in R], float)
    n = len(R); d = B - A
    rng = np.random.default_rng(0)
    bs_ = [d[rng.integers(0, n, n)].mean() for _ in range(3000)]
    lo, hi = np.percentile(bs_, [2.5, 97.5])
    nb, na = int(((B == 1) & (A == 0)).sum()), int(((A == 1) & (B == 0)).sum())
    from scipy.stats import binomtest
    p = binomtest(nb, nb + na, 0.5).pvalue if nb + na else 1.0
    print(f"試行 {n}（{len({r['i'] for r in R})} 配置 × 部品）")
    print(f"A 触らずに掴む（SmolVLM で選んでそのまま）: {A.mean() * 100:.1f}%")
    print(f"B 1回触って確かめてから掴む（SmolVLM＋提案手法）: {B.mean() * 100:.1f}%")
    print(f"差 B−A: {d.mean() * 100:+.1f} ポイント（95% 信頼区間 [{lo * 100:+.1f}, {hi * 100:+.1f}]）、"
          f"片方だけ成功 B {nb}・A {na}、McNemar の正確検定 p = {p:.3g}")
    for v in sorted({r["vis"] for r in R}):
        sel = [k for k, r in enumerate(R) if r["vis"] == v]
        print(f"  見え方 {v:.0f}mm（{len(sel)} 回）: A {A[sel].mean() * 100:.0f}%  B {B[sel].mean() * 100:.0f}%")
    for sl in (0, 1):
        sel = [k for k, r in enumerate(R) if r["slot"] == sl]
        print(f"  部品 {sl + 1}（{len(sel)} 回）: A {A[sel].mean() * 100:.0f}%  B {B[sel].mean() * 100:.0f}%")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(); sys.exit()
    k, n, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out_dir = os.environ.get("RESULTS_DIR", "results"); os.makedirs(out_dir, exist_ok=True)
    fn = f"{out_dir}/vlmcmp_{k}.jsonl"
    done = {(json.loads(l)["i"], json.loads(l)["slot"]) for l in open(fn)} if os.path.exists(fn) else set()
    out = open(fn, "a")
    jobs = [(i, sl) for i in range(N) for sl in range(len(VP.SLOTS))]
    for j, (i, sl) in enumerate(jobs):
        if j % n != k or (i, sl) in done:
            continue
        out.write(json.dumps(trial(i, sl)) + "\n"); out.flush()
