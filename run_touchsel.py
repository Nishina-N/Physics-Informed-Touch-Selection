"""触る点の選び方の比較：同じ配置で、1回目・2回目に触る点を3通りの規則で選び、その時点で掴んだ結果を比べる。
  voi : 触った後の最良の成功確率の期待値が最大の点（本研究）
  unc : 部品がそこにあるかどうかの確率 p の分散 p(1−p) が最大の点（Act-VH と同じ「形の不確かさが最大」の考え方）
  rand: 候補から一様に選ぶ
使い方: python3 run_touchsel.py k n N predictor.pkl → results/tsel_k.jsonl
"""
import sys, gc, json, os, time
import numpy as np
import shelf as S
import perception as P
import partfilter as F
import build_scene as bs
import run_branch as RB

RULES = ("voi", "unc", "rand")
N_TOUCH = 2


def occupancy(parts, x, y):
    """点 (x,y) の真下に本体があると考える粒子の割合。"""
    a, b = F.axial(parts, x, y)
    return ((a >= parts[:, 3]) & (a <= parts[:, 3] + parts[:, 4]) & (np.abs(b) <= F.W / 2)).mean()


def choose(rule, est, gc_, tc, rng):
    if rule == "voi":
        return F.choose_touch_m(est, gc_, tc)[0]
    if rule == "unc":
        v = [occupancy(est, x, y) for x, y in tc]
        v = [p * (1 - p) for p in v]
        return tc[int(np.argmax(v))]
    return tc[rng.integers(len(tc))]


def trial(i, rule, pred):
    lay = RB.layout(i); rng = np.random.default_rng(5000 + i); rr = np.random.default_rng(9000 + i)
    s = S.ShelfSim(RB.W, visible=lay["vis"], part=(lay["fl"], lay["L"]), dxy=(0, lay["dy"]), yaw_deg=lay["yaw"])
    RB.apply_phys(s, lay); F.use_predictor(pred, s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    est, _ = P.init_particles_part(cam, cm, occl, RB.W, ztab, rng, (S.X_FRONT, S.Y0), RB.FL_R, RB.L_R, yaw0=yaw0)
    est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, RB.FL_R, rng)
    R = d.body(s.cube).xmat.reshape(3, 3)
    front = d.body(s.cube).xpos + R @ np.array([-(lay["fl"] + lay["L"] / 2), 0, 0])
    steps = []
    for k in range(1, N_TOUCH + 1):
        gc_ = F.axis_points(est); tc = F.axis_points(est, step=0.003)
        tx, ty = choose(rule, est, gc_, tc, rr)
        r = s.touch([tx, ty])
        est = F.update(est, tx, ty, r["obs"], rng)
        (gx, gy), p = F.best_grasp_m(est, F.axis_points(est))
        res = RB.branch_grasp(s, [gx, gy, ztab + RB.W / 2])
        steps.append(dict(k=k, touch_rel_mm=float((tx - front[0]) * 1000), obs=r["obs"], p=p, **res,
                          touch_moved_total_mm=r["moved_total_mm"],
                          body_front_sd_mm=float((est[:, 0] + est[:, 3]).std() * 1000)))
    out = dict(i=i, rule=rule, vis=lay["vis"] * 1000, body_mm=[lay["fl"] * 1000, (lay["fl"] + lay["L"]) * 1000],
               body_seen=seen, steps=steps)
    del s; gc.collect()
    return out


if __name__ == "__main__":
    k, n, N, pred = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out_dir = os.environ.get("RESULTS_DIR", "results")             # 結果の置き場所（既定 results）
    os.makedirs(out_dir, exist_ok=True)
    fn = f"{out_dir}/tsel_{k}.jsonl"
    done = {(json.loads(l)["i"], json.loads(l)["rule"]) for l in open(fn)} if os.path.exists(fn) else set()
    out = open(fn, "a")
    jobs = [(i, r) for i in range(N) for r in RULES]
    for j, (i, r) in enumerate(jobs):
        if j % n != k or (i, r) in done:
            continue
        out.write(json.dumps(trial(i, r, pred)) + "\n"); out.flush()
