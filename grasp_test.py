"""把持の成立性テスト（10/1のハンド判断用）。
中指と親指の指先で立方体を挟み、5cm持ち上げて2秒保持できるかを試す。
- 腕・手には重力補償を入れる（実機の低レベル制御のフィードフォワード相当）
- 指は「開→閉」の関節空間の直線軌道 s∈[0,1] 上を動かし、両指先が触れたら少しだけ押し込んで止める
"""
import argparse
import numpy as np
import mujoco
from build_scene import build

ARM = [f"right_{j}_joint" for j in
       ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]]
HAND = [f"right_hand_{j}_joint" for j in
        ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"]]

# 指の軌道：(thumb_0, thumb_1, thumb_2, middle_0, middle_1)。人差し指は伸ばしたまま使わない
Q_OPEN = np.array([0.30, 0.50, 0.0, 0.0, 0.0])
Q_CLOSE = np.array([0.30, -0.263, -1.167, 0.698, 1.556])  # 親指先と中指先がちょうど触れる姿勢（候補4）
Q_LO = np.array([-1.05, -1.05, -1.75, 0.0, 0.0])
Q_HI = np.array([1.05, 0.72, 0.0, 1.57, 1.75])
S_END = 1.15   # 閉じ指令の終点（触れる姿勢の少し先）

# 手首姿勢：手首x軸まわりに-75°回し、指先を20°下げる
ROLL_DEG, PITCH_DEG = -75.0, 20.0
# 開始時の構え：挟む点をここに置く。頭部カメラから机上の立方体が隠れない位置（見え率99.8%を確認）
READY = np.array([0.15, -0.40, 1.10])

# 立方体の大きさごとの設定：接近時の指の開き s0、手首座標で見た「挟む点」、両指先が触れてからの押し込み量 squeeze（s）
# 挟む点＝軌道上で指先の隙間が立方体の幅になる時の、指先どうしの最近接点の中点
PER_SIZE = {
    "cube_20mm": dict(s0=0.75, squeeze=0.03, grasp_pt=np.array([0.126, 0.072, -0.021])),  # s=0.90で隙間19mm
    "cube_50mm": dict(s0=0.45, squeeze=0.08, grasp_pt=np.array([0.128, 0.076, -0.017])),  # s=0.75で隙間50mm
}


def hand_q(s):
    t0, t1, t2, m0, m1 = np.clip(Q_OPEN + s * (Q_CLOSE - Q_OPEN), Q_LO, Q_HI)
    return dict(zip(HAND, [t0, t1, t2, 0.0, 0.0, m0, m1]))


def r_target():
    r, p = np.radians(ROLL_DEG), np.radians(PITCH_DEG)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return Ry @ Rx   # 列＝手首x,y,zの世界での向き（手首x＝指の向き、手首y＝手のひら）


class Sim:
    def __init__(self, cube, seed=0, pos_noise=0.0, record=False):
        self.m = build().compile()
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.frames = None
        if record:                                     # 録画：30fpsで横から＋頭部カメラ
            self.every = max(1, round(1 / 30 / m.opt.timestep))
            self.renderer = mujoco.Renderer(m, 480, 640)
            self.seg = mujoco.Renderer(m, 480, 640)       # 頭部カメラでの物体の見え方（画素数）を数える用
            self.seg.enable_segmentation_rendering()
            self.cam = mujoco.MjvCamera()
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.35, 240, -10
        self.cube = cube
        self.grasp_pt = PER_SIZE[cube]["grasp_pt"]
        rng = np.random.default_rng(seed)
        key = m.key("stand")
        n = m.nq - 14                      # 末尾14は立方体2個の自由関節
        d.qpos[:n] = key.qpos[:n]
        d.ctrl[:] = key.ctrl
        other = "cube_20mm" if cube == "cube_50mm" else "cube_50mm"
        self._set_cube(other, [0.70, -0.45])           # 使わない立方体は机の端へ退避
        self._set_cube(cube, np.array([0.30, -0.18]) + rng.uniform(-pos_noise, pos_noise, 2))
        if record:
            self.cam.lookat[:] = [*d.qpos[m.joint(cube + "_free").qposadr[0]:][:2], 0.84]
        self.arm_q = [m.joint(j).qposadr[0] for j in ARM]
        self.arm_v = [m.joint(j).dofadr[0] for j in ARM]
        self.gc_v = self.arm_v + [m.joint(j).dofadr[0] for j in HAND]
        self.act = {m.actuator(i).name: i for i in range(m.nu)}
        mujoco.mj_forward(m, d)
        q_ready, _ = self.ik(READY)                    # 構えの姿勢から始める（カメラの視界を空ける）
        for j, a, q in zip(ARM, self.arm_q, q_ready):
            d.qpos[a] = q
            d.ctrl[self.act[j]] = q
        self.set_hand(hand_q(PER_SIZE[cube]["s0"]))
        mujoco.mj_forward(m, d)
        self.step(0.5)
        if record:                                     # 姿勢が落ち着いてから録画開始
            self.frames, self.nstep = [], 0

    def _set_cube(self, name, xy):
        j = self.m.joint(name + "_free")
        top = self.m.geom("table").size[2] * 2
        s = self.m.geom(name).size[0]
        self.d.qpos[j.qposadr[0]:j.qposadr[0] + 7] = [xy[0], xy[1], top + s + 0.001, 1, 0, 0, 0]

    def set_hand(self, targets):
        for j, v in targets.items():
            self.d.ctrl[self.act[j]] = v

    def mj_step(self):
        self.d.qfrc_applied[self.gc_v] = self.d.qfrc_bias[self.gc_v]   # 右腕・右手の重力補償
        mujoco.mj_step(self.m, self.d)
        if self.frames is not None:
            self.nstep += 1
            if self.nstep % self.every == 0:
                self.frames.append(self._frame())

    def _frame(self):
        """左：横から見た映像、右：頭部カメラ映像（物体が見えている画素数を表示）。"""
        from PIL import Image, ImageDraw
        m, d = self.m, self.d
        self.renderer.update_scene(d, self.cam); side = self.renderer.render()
        self.renderer.update_scene(d, "head_cam"); head = self.renderer.render()
        gid = m.geom(self.cube).id
        def cube_px(opt=None):
            self.seg.update_scene(d, "head_cam", scene_option=opt); ids = self.seg.render()
            return int(((ids[..., 0] == gid) & (ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)).sum())
        vis = cube_px()                                  # ロボットに隠された状態で見えている画素
        no_robot = mujoco.MjvOption(); no_robot.geomgroup[:] = [1, 0, 0, 0, 0, 0]   # ロボット(グループ2,3)を消す
        full = max(cube_px(no_robot), 1)                 # 隠れがないときの画素＝見え率の分母
        img = Image.fromarray(np.hstack([side, head])); dr = ImageDraw.Draw(img)
        dr.text((10, 10), "side view", fill=(255, 255, 255))
        dr.text((650, 10), f"head camera  cube visible: {vis}/{full} px ({100 * vis / full:.0f}%)", fill=(255, 255, 0))
        return np.array(img)

    def step(self, seconds):
        for _ in range(int(seconds / self.m.opt.timestep)):
            self.mj_step()

    def ik(self, p_target, iters=300):
        """挟む点を p_target に、手首を r_target() に合わせる右腕7関節（減衰最小二乗）。"""
        m, d = self.m, mujoco.MjData(self.m)
        d.qpos[:] = self.d.qpos
        bid = m.body("right_wrist_yaw_link").id
        RT = r_target()
        for _ in range(iters):
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            R = d.xmat[bid].reshape(3, 3)
            p = d.xpos[bid] + R @ self.grasp_pt
            e_p = p_target - p
            e_r = 0.5 * sum(np.cross(R[:, i], RT[:, i]) for i in range(3))
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, p, bid)
            J = np.vstack([jp[:, self.arm_v], jr[:, self.arm_v]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), np.concatenate([e_p, e_r]))
            dq = np.clip(dq, -0.1, 0.1)
            for k, (j, a) in enumerate(zip(ARM, self.arm_q)):
                lo, hi = m.joint(j).range
                d.qpos[a] = np.clip(d.qpos[a] + dq[k], lo, hi)
            if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 1e-2:
                break
        return np.array([d.qpos[a] for a in self.arm_q]), np.linalg.norm(e_p)

    def move_arm(self, p_target, seconds=1.0):
        q_goal, err = self.ik(p_target)
        q0 = np.array([self.d.ctrl[self.act[j]] for j in ARM])
        n = int(seconds / self.m.opt.timestep)
        for i in range(n):
            s = (i + 1) / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5          # 最小躍度の補間
            for k, j in enumerate(ARM):
                self.d.ctrl[self.act[j]] = q0[k] + s * (q_goal[k] - q0[k])
            self.mj_step()
        return err

    def close_until_contact(self, seconds=1.0):
        """s0→S_END へ閉じ、親指先・中指先の両方が物体に触れたら s+SQUEEZE で止める。"""
        m, d = self.m, self.d
        cid = m.body(self.cube).id
        tips = {m.body("right_hand_thumb_2_link").id, m.body("right_hand_middle_1_link").id}
        s0 = PER_SIZE[self.cube]["s0"]
        n = int(seconds / m.opt.timestep)
        hold = None
        for i in range(n):
            s = s0 + (S_END - s0) * (i + 1) / n
            if hold is None:
                touch = set()
                for c in d.contact[:d.ncon]:
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    if cid in (b1, b2):
                        touch.add(b2 if b1 == cid else b1)
                if tips <= touch:
                    hold = s + PER_SIZE[self.cube]["squeeze"]
            self.set_hand(hand_q(hold if hold is not None else s))
            self.mj_step()
        return hold


def trial(cube, seed, pos_noise, verbose=False, record=False):
    sim = Sim(cube, seed, pos_noise, record)
    c0 = sim.d.body(cube).xpos.copy()   # 決め打ち：真の位置を使う（成立性の確認なので推定誤差なし）
    errs = [sim.move_arm(c0 + [0, 0, 0.10], 1.5),   # 上空へ
            sim.move_arm(c0, 1.0)]                  # 挟む点を立方体中心へ
    hold = sim.close_until_contact(1.0); sim.step(0.3)
    errs.append(sim.move_arm(c0 + [0, 0, 0.05], 1.0))  # 5cm持ち上げ
    sim.step(2.0)                                      # 2秒保持
    rise = sim.d.body(cube).xpos[2] - c0[2]
    ok = rise > 0.04
    if verbose:
        h = "未接触" if hold is None else f"s={hold:.2f}で保持"
        print(f"{cube} seed={seed} 上昇{rise*100:.1f}cm {h} IK誤差最大{max(errs)*1000:.1f}mm {'成功' if ok else '失敗'}")
    return ok, sim


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.02, help="置き位置のばらつき[m]")
    ap.add_argument("--video", action="store_true", help="各立方体1試行を録画して grasp_<cube>.mp4 に保存")
    a = ap.parse_args()
    if a.video:
        import imageio.v2 as imageio
        for cube in ["cube_50mm", "cube_20mm"]:
            ok, sim = trial(cube, 0, a.noise, verbose=True, record=True)
            imageio.mimsave(f"grasp_{cube}.mp4", sim.frames, fps=30)
            imageio.imwrite(f"grasp_{cube}_lift.png", sim.frames[-1])   # 保持中の最後のコマ
            print(f"保存: grasp_{cube}.mp4, grasp_{cube}_lift.png")
        raise SystemExit
    for cube in ["cube_50mm", "cube_20mm"]:
        res = [trial(cube, s, a.noise, verbose=True)[0] for s in range(a.n)]
        print(f"== {cube}: {sum(res)}/{a.n}")
