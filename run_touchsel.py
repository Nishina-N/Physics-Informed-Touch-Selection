"""触る点の選び方の比較：同じ配置で、1回目・2回目に触る点を3通りの規則で選び、その時点で掴んだ結果を比べる。
  voi : 触った後の最良の成功確率の期待値が最大の点（本研究）
  unc : 部品がそこにあるかどうかの確率 p の分散 p(1−p) が最大の点（Act-VH と同じ「形の不確かさが最大」の考え方）
  rand: 候補から一様に選ぶ
  geom: voi と同じ先読みだが、成功確率に学習した予測器を使わず、幾何の規則（本体の縁から余裕の内側なら成功）を使う。
        触る点も掴む点もこの規則で選ぶ（簡単な見積もりと細かい見積もりの比較）
  fixshape: voi と同じ先読みと予測器だが、形（フランジと本体の長さ）は範囲の中央の値に決め打ちし、位置と向きだけを
        推定・更新する（DA-GRD のように形を既知とする条件。隠れた形を推定することの価値を見る比較）
使い方: python3 run_touchsel.py k n N predictor.pkl → results/tsel_k.jsonl（voi・unc・rand）
        TSEL_RULES=geom python3 run_touchsel.py k n N predictor.pkl → results/tselgeom_k.jsonl
        TSEL_RULES=fixshape python3 run_touchsel.py k n N predictor.pkl → results/tselfixshape_k.jsonl
"""
import sys, gc, json, os, time
import numpy as np
import shelf as S
import perception as P
import partfilter as F
import build_scene as bs
import run_branch as RB

FIX_SHAPE = (np.mean(RB.FL_R), np.mean(RB.L_R))   # fixshape の決め打ちの形（範囲の中央：フランジ 20mm、本体 16mm）
RULES = tuple(os.environ.get("TSEL_RULES", "voi,unc,rand").split(","))
N_TOUCH = 2


def occupancy(parts, x, y):
    """点 (x,y) の真下に本体があると考える粒子の割合。"""
    a, b = F.axial(parts, x, y)
    return ((a >= parts[:, 3]) & (a <= parts[:, 3] + parts[:, 4]) & (np.abs(b) <= F.W / 2)).mean()


def choose(rule, est, gc_, tc, rng):
    if rule in ("voi", "geom", "fixshape"):                     # geom は予測器を読み込まないので、幾何の規則で同じ先読みをする
        return F.choose_touch_m(est, gc_, tc)[0]
    if rule == "unc":
        v = [occupancy(est, x, y) for x, y in tc]
        v = [p * (1 - p) for p in v]
        return tc[int(np.argmax(v))]
    return tc[rng.integers(len(tc))]


def trial(i, rule, pred):
    lay = RB.layout(i); rng = np.random.default_rng(5000 + i); rr = np.random.default_rng(9000 + i)
    s = S.ShelfSim(RB.W, visible=lay["vis"], part=(lay["fl"], lay["L"]), dxy=(0, lay["dy"]), yaw_deg=lay["yaw"])
    RB.apply_phys(s, lay)
    if rule == "geom":
        F._PRED = None                                # 成功確率は幾何の規則（partfilter.prob_matrix）
    else:
        F.use_predictor(pred, s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    flr, lr = ((FIX_SHAPE[0],) * 2, (FIX_SHAPE[1],) * 2) if rule == "fixshape" else (RB.FL_R, RB.L_R)   # fixshape は決め打ちの形で照合する
    est, _ = P.init_particles_part(cam, cm, occl, RB.W, ztab, rng, (S.X_FRONT, S.Y0), flr, lr, yaw0=yaw0)
    if rule == "fixshape":                            # 形は決め打ち（深度で本体の縁を探すのも形の推定なので行わない）
        est[:, 3], est[:, 4] = FIX_SHAPE; seen = None
    else:
        est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, RB.FL_R, rng)
    R = d.body(s.cube).xmat.reshape(3, 3)
    front = d.body(s.cube).xpos + R @ np.array([-(lay["fl"] + lay["L"] / 2), 0, 0])
    steps = []
    for k in range(1, N_TOUCH + 1):
        gc_ = F.axis_points(est); tc = F.axis_points(est, step=0.003)
        tx, ty = choose(rule, est, gc_, tc, rr)
        r = s.touch([tx, ty])
        est = F.update(est, tx, ty, r["obs"], rng)
        if rule == "fixshape":                        # 更新で加わる長さのゆらぎも戻す
            est[:, 3], est[:, 4] = FIX_SHAPE
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
    tag = "tsel" if RULES == ("voi", "unc", "rand") else "tsel" + "".join(RULES)
    fn = f"{out_dir}/{tag}_{k}.jsonl"
    done = {(json.loads(l)["i"], json.loads(l)["rule"]) for l in open(fn)} if os.path.exists(fn) else set()
    out = open(fn, "a")
    jobs = [(i, r) for i in range(N) for r in RULES]
    for j, (i, r) in enumerate(jobs):
        if j % n != k or (i, r) in done:
            continue
        out.write(json.dumps(trial(i, r, pred)) + "\n"); out.flush()
