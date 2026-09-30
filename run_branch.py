"""本実験（分岐評価）：1つの配置で、触る→（その時点で掴んでみる：状態を保存して掴み、戻す）を最大 MAX_TOUCH 回繰り返す。
触る点の選び方は条件によらないので、k回触った時点で掴んだ結果から A（k=0）・Bn（k=n）・C（成功確率が初めてτ以上になった k）
を同じ試行から計算できる。
使い方: python3 run_branch.py k n N [predictor.pkl] → results/branch_k.jsonl
"""
import sys, gc, json, os, time
import numpy as np
import mujoco
import shelf as S
import perception as P
import partfilter as F
import build_scene as bs

P.TEMP = 0.05
W = 0.020
FL_R, L_R = (0.010, 0.030), (0.012, 0.020)
VIS_LEVELS = (0.010, 0.015, 0.020, 0.025)
MAX_TOUCH = 4
MU_R, MASS_R, GAIN_R = (0.3, 1.0), (0.7, 1.3), (0.8, 1.2)


def layout(i):
    rng = np.random.default_rng(1000 + i)
    return dict(vis=VIS_LEVELS[i % 4], fl=rng.uniform(*FL_R), L=rng.uniform(*L_R),
                dy=rng.uniform(-0.01, 0.01), yaw=rng.uniform(-20, 20),
                mu=rng.uniform(*MU_R), mass=rng.uniform(*MASS_R), gain=rng.uniform(*GAIN_R))


def apply_phys(s, lay):
    m = s.m
    hand = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith("right_hand")]
    for gid in hand + [m.geom(s.cube).id, m.geom(s.cube + "_flange").id]:
        m.geom_friction[gid, 0] = lay["mu"]
    b = m.body(s.cube).id; m.body_mass[b] *= lay["mass"]; m.body_inertia[b] *= lay["mass"]
    for i in range(m.nu):
        if m.actuator(i).name.startswith("right_hand"):
            m.actuator_gainprm[i, 0] *= lay["gain"]; m.actuator_biasprm[i, 1] *= lay["gain"]
    mujoco.mj_setConst(m, mujoco.MjData(m)); mujoco.mj_forward(m, s.d)   # 質量を変えたら定数を計算し直す（しないと不安定になり、成功率が見かけ上下がった）。s.d を作業領域に使うと状態が壊れる


def branch_grasp(s, g):
    """今のシミュレーションの状態を複製し、複製の上で掴んで結果を返す（元の状態には触れない）。
    mj_setState で戻す方法は、戻した直後の掴みが不安定になった（慣性行列が特異に近いという警告）。"""
    import copy
    d0 = s.d
    dc = copy.copy(d0)
    yaw_saved, hits = S.YAW_DEG, set(s.board_hits)
    s.d = s.sim.d = dc
    before = dc.body(s.cube).xpos.copy()
    r = s.grasp(g)
    moved = float(np.linalg.norm(dc.body(s.cube).xpos[:2] - before[:2]) * 1000)
    s.d = s.sim.d = d0
    S.YAW_DEG = yaw_saved; s.board_hits = hits
    del dc
    return dict(success=r["success"], moved_mm=None if r["success"] else moved, board=r["board"])


def trial(i, pred=None):
    lay = layout(i); rng = np.random.default_rng(5000 + i); t0 = time.time()
    s = S.ShelfSim(W, visible=lay["vis"], part=(lay["fl"], lay["L"]), dxy=(0, lay["dy"]), yaw_deg=lay["yaw"])
    apply_phys(s, lay)
    if pred:
        F.use_predictor(pred, s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    est, _ = P.init_particles_part(cam, cm, occl, W, ztab, rng, (S.X_FRONT, S.Y0), FL_R, L_R, yaw0=yaw0)
    est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, FL_R, rng)
    R = d.body(s.cube).xmat.reshape(3, 3); yw = np.arctan2(R[1, 0], R[0, 0])
    front = d.body(s.cube).xpos + R @ np.array([-(lay["fl"] + lay["L"] / 2), 0, 0])
    steps = []
    for k in range(MAX_TOUCH + 1):
        gc_ = F.axis_points(est)
        (gx, gy), p = F.best_grasp_m(est, gc_)
        res = branch_grasp(s, [gx, gy, ztab + W / 2])
        a = float(np.cos(yw) * (gx - front[0]) + np.sin(yw) * (gy - front[1]))
        b = float(-np.sin(yw) * (gx - front[0]) + np.cos(yw) * (gy - front[1]))
        step = dict(k=k, p=p, grasp_a_mm=a * 1000, grasp_b_mm=b * 1000, **res)
        if k < MAX_TOUCH:
            (tx, ty), ev = F.choose_touch_m(est, gc_, F.axis_points(est, step=0.003))
            r = s.touch([tx, ty])
            est = F.update(est, tx, ty, r["obs"], rng)
            step.update(touch_rel_mm=float((tx - front[0]) * 1000), obs=r["obs"], ev=ev, touch_moved_mm=r["moved_mm"])
        steps.append(step)
    out = dict(i=i, **{k: (v * 1000 if k in ("vis", "fl", "L", "dy") else v) for k, v in lay.items()},
               body_seen=seen, body_mm=[lay["fl"] * 1000, (lay["fl"] + lay["L"]) * 1000], steps=steps,
               sec=time.time() - t0)
    del s; gc.collect()
    return out


if __name__ == "__main__":
    k, n, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    pred = sys.argv[4] if len(sys.argv) > 4 else None
    os.makedirs("results", exist_ok=True)
    tag = "branch" if pred else "branch_geom"
    fn = f"results/{tag}_{k}.jsonl"
    done = {json.loads(l)["i"] for l in open(fn)} if os.path.exists(fn) else set()
    out = open(fn, "a")
    for i in range(N):
        if i % n != k or i in done:
            continue
        out.write(json.dumps(trial(i, pred)) + "\n"); out.flush()
