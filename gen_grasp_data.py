"""予測器の学習データ：形の候補ごとの成功確率を学ぶため、真の形に対する挟む点を振って掴み、成否を記録する。
物理のばらつき（摩擦・質量・指の制御ゲイン）は毎回引き、特徴量には入れない。
使い方: python3 gen_grasp_data.py k n N → results/gdata_k.jsonl
"""
import sys, gc, json, os, time
import numpy as np
import shelf as S
import build_scene as bs
import mujoco                     # shelf の後に読み込む（描画の設定を先に決める）

W = 0.020
FL_R, L_R = (0.010, 0.030), (0.012, 0.020)
VIS_LEVELS = (0.010, 0.015, 0.020, 0.025)
MU_R, MASS_R, GAIN_R = (0.3, 1.0), (0.7, 1.3), (0.8, 1.2)


def randomize(s, rng):
    m = s.m
    mu = rng.uniform(*MU_R)
    hand = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith("right_hand")]
    for gid in hand + [m.geom(s.cube).id, m.geom(s.cube + "_flange").id]:
        m.geom_friction[gid, 0] = mu              # 接触の摩擦は2つの形状の大きい方なので、指と部品の両方に入れる
    ms = rng.uniform(*MASS_R)
    b = m.body(s.cube).id
    m.body_mass[b] *= ms; m.body_inertia[b] *= ms
    gs = rng.uniform(*GAIN_R)
    for i in range(m.nu):
        if m.actuator(i).name.startswith("right_hand"):
            m.actuator_gainprm[i, 0] *= gs; m.actuator_biasprm[i, 1] *= gs
    mujoco.mj_setConst(m, mujoco.MjData(m)); mujoco.mj_forward(m, s.d)   # 質量を変えたら定数を計算し直す（しないと不安定になり、成功率が見かけ上下がった）。s.d を作業領域に使うと状態が壊れる
    return dict(mu=mu, mass_scale=ms, gain_scale=gs)


def sample(i):
    rng = np.random.default_rng(20000 + i)
    fl, L = rng.uniform(*FL_R), rng.uniform(*L_R)
    vis = VIS_LEVELS[rng.integers(4)]
    yaw, dy = rng.uniform(-20, 20), rng.uniform(-0.01, 0.01)
    t0 = time.time()
    s = S.ShelfSim(W, visible=vis, part=(fl, L), dxy=(0, dy), yaw_deg=yaw)
    phys = randomize(s, rng)
    d = s.d
    R = d.body(s.cube).xmat.reshape(3, 3); yw = np.arctan2(R[1, 0], R[0, 0])
    front = d.body(s.cube).xpos + R @ np.array([-(fl + L / 2), 0, 0])
    a = rng.uniform(fl - 0.006, fl + L + 0.004)          # 部品の軸に沿った挟む点（手前の面から）
    b = rng.uniform(-0.008, 0.008)                        # 横のずれ
    gx = front[0] + np.cos(yw) * a - np.sin(yw) * b
    gy = front[1] + np.sin(yw) * a + np.cos(yw) * b
    gz = bs.TABLE_TOP_Z + W / 2 + rng.normal(0, 0.001)
    g = s.grasp([gx, gy, gz])
    rec = dict(i=i, fl=fl, L=L, vis=vis, yaw=float(np.degrees(yw)), dy=dy, a=a, b=b, dz=gz - bs.TABLE_TOP_Z - W / 2,
               u=a - fl, v=fl + L - a, reach=gx - s.front_x, success=g["success"], closed=g["closed"],
               board=g["board"], sec=time.time() - t0, **phys)
    del s; gc.collect()
    return rec


if __name__ == "__main__":
    k, n, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("results", exist_ok=True)
    fn = f"results/gdata_{k}.jsonl"
    done = set()
    if os.path.exists(fn):
        done = {json.loads(l)["i"] for l in open(fn)}
    out = open(fn, "a")
    for i in range(N):
        if i % n != k or i in done:
            continue
        out.write(json.dumps(sample(i)) + "\n"); out.flush()
