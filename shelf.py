"""棚の下の小物体を扱う共通部品：シーン、中指で上から触る、挟んで掴む。

- 物体は前方 X0・右 Y0（右腕の到達可能範囲の地図から決めた位置）。天板は机上20cm（build_scene.SHELF_H）。
- 見え方は visible（フランジの上面が手前から見える長さ[m]）で指定し、それに合わせて天板の前端を置く。
- 手首は手首x軸まわり−60°・指先10°下げ、鉛直軸まわり−55°。中指と親指の指先でつまみ、人差し指は握り込む。
- IK は mink（右腕7関節）。
"""
import numpy as np
import build_scene as bs          # 描画の設定（MUJOCO_GL）を mujoco より先に決めるため、最初に読み込む
import mujoco
import grasp_test as g
import ik_mink

X0, Y0 = 0.32, -0.25
X_FRONT = 0.31          # 直方体の手前の面のx[m]（2cm立方体の中心がX0に来る位置）
CAM_P = np.array([0.0540, 0.0173, 1.2668])   # 開始姿勢の頭部カメラの位置（シムで実測。見える長さの計算に使う）


def board_front_for_visible(visible, width):
    """上面が手前の面から visible[m] 先まで見えるような天板の前端のx。
    頭部カメラから天板の前端の下の角を通る光線が、物体の上面の高さに届く点が見える範囲の奥の端になる。"""
    zb = bs.TABLE_TOP_Z + bs.SHELF_H; zt = bs.TABLE_TOP_Z + width
    x_occ = X_FRONT + visible
    return CAM_P[0] + (x_occ - CAM_P[0]) * (CAM_P[2] - zb) / (CAM_P[2] - zt)
PASS_H = 0.020          # 差し込むときの、挟む点の物体中心からの高さ[m]（手首−55°で天板と約1cm空く）
TOUCH_F = 0.1           # 触ったと判定する接触力[N]（約10g）
TOUCH_FLOOR = -0.001    # この高さ（机上）まで下ろす[m]。机に0.1N当たれば外れ（2mmで止めると高さ4mmのフランジに力が乗らなかった）
TOUCH_YAW = -55.0       # 触るときの手首の鉛直軸まわりの角度[deg]。掴むときと同じ（持ち替えなし）
MIDDLE_TOUCH = None     # 触る間の中指（middle_0, middle_1）。Noneなら掴むときの開きのまま
TOUCH_START = 0.010     # 触り始める指先の高さ（物体の上面から）[m]
TIP_CENTER = np.array([0.0507, -0.0040, -0.0062])   # 中指の先の中央（指のリンク座標）。物体の真上を触ったときの接触点
TIP_RADIUS = 0.004      # 接触点が指先の中央からこの距離以内なら「指先の中央で当たり」[m]
THUMB_UP = (0.72, 0.0)
LEFT_ARM = {"left_shoulder_roll_joint": 0.5, "left_elbow_joint": 1.2, "left_wrist_pitch_joint": 0.0}  # 触る間の親指（thumb_1, thumb_2）。中指の先より約33mm上に退避

g.ROLL_DEG, g.PITCH_DEG = -60.0, 10.0
GRASP_YAW = -55.0       # 掴むときの手首の鉛直軸まわりの角度[deg]
YAW_DEG = -55.0         # 手首を鉛直軸まわりに回す[deg]。挟む軸を左右方向（物体の面にほぼ垂直、水平角−86°）にする
TILT_DEG = 0.0          # 手首を奥行き軸まわりに回す[deg]。6°で挟む軸が水平になるが、天板との隙間がなくなるので0
_r_target = g.r_target
def _r_target_yaw():
    y = np.radians(YAW_DEG)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    t = np.radians(TILT_DEG)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    return Rx @ Rz @ _r_target()
g.r_target = _r_target_yaw
g.PER_SIZE["cube_15mm"] = dict(s0=0.75, squeeze=0.03, grasp_pt=np.array([0.1248, 0.0702, -0.0213]))
_hand_q = g.hand_q
def _hand_q_index_curled(s):
    q = _hand_q(s)
    q["right_hand_index_0_joint"], q["right_hand_index_1_joint"] = 1.3, 1.6   # 人差し指は握り込む
    return q
g.hand_q = _hand_q_index_curled
ik_mink.use_mink()


def hand_q2(s_thumb, s_middle):
    """親指と中指を別々の進み具合で閉じる（先に触れた指はそこで止め、もう一方だけ進める）。"""
    qt = _hand_q_index_curled(s_thumb); qm = _hand_q_index_curled(s_middle)
    for j in ["right_hand_middle_0_joint", "right_hand_middle_1_joint"]:
        qt[j] = qm[j]
    return qt


def close_each(sim, seconds=1.0, squeeze=None):
    """各指を開→閉へ進め、物体に触れた指はそこで止める。両方触れたら、両方を squeeze だけ同時に進める。"""
    m, d = sim.m, sim.d
    cid = m.body(sim.cube).id
    tips = {"thumb": m.body("right_hand_thumb_2_link").id, "middle": m.body("right_hand_middle_1_link").id}
    s0 = g.PER_SIZE[sim.cube]["s0"]
    sq = g.PER_SIZE[sim.cube]["squeeze"] if squeeze is None else squeeze
    n = int(seconds / m.opt.timestep); ds = (g.S_END - s0) / n
    s = {"thumb": s0, "middle": s0}; stop = {}
    for _ in range(n):
        touch = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if cid in (b1, b2):
                touch.add(b2 if b1 == cid else b1)
        for f in s:
            if f not in stop and tips[f] in touch:
                stop[f] = s[f]
            if f not in stop:
                s[f] = min(g.S_END, s[f] + ds)
        if len(stop) == 2:
            break
        sim.set_hand(hand_q2(s["thumb"], s["middle"])); sim.mj_step()
    if len(stop) < 2:
        return None
    k = max(1, int(0.3 / m.opt.timestep))                  # 0.3秒かけて両方を同時に締める
    for i in range(k):
        a = sq * (i + 1) / k
        sim.set_hand(hand_q2(stop["thumb"] + a, stop["middle"] + a)); sim.mj_step()
    return stop


def cube_name(size):
    return f"cube_{int(round(size * 1000))}mm"


class ShelfSim:
    """棚の下に立方体を1個置いた試行1回分。"""

    def __init__(self, size, depth=None, seed=0, dxy=(0.0, 0.0), yaw_deg=0.0, record=False, length=None, visible=None,
                 part=None):
        """size: 幅（＝高さ）。length: 奥行き方向の長さ（Noneなら立方体）。
        visible を与えると、手前の面を X_FRONT に置き、上面が手前から visible[m] 先まで見えるよう天板を置く。
        与えなければ従来どおり、中心を X0 に置き、天板の前端を中心から depth 手前に置く。
        part=(flange_len, body_len)：手前にフランジ（高さ bs.FLANGE_H）、奥に本体をつないだ部品。
        visible はフランジの上面が見える長さ。剛体の原点（c0）は本体の中心。"""
        self.part = part
        if part is not None:
            self.flange_len, length = part
        self.size, self.length = size, (size if length is None else length)
        if visible is not None:
            top_h = bs.FLANGE_H if part is not None else size
            self.front_x = board_front_for_visible(visible, top_h)
            cx = X_FRONT + (part[0] if part is not None else 0.0) + self.length / 2
        else:
            self.front_x = X0 - depth; cx = X0
        self.depth, self.visible = cx - self.front_x, visible
        g.build = lambda cube_sizes=None: bs.build_shelf(size, self.front_x, Y0, length=length, part=part)
        self.cube = cube_name(size)
        self.sim = g.Sim(self.cube, seed, 0.0, record)
        m, d = self.m, self.d = self.sim.m, self.sim.d
        # 重力補償をロボット全体に広げる（右腕だけだと腰が約4°たわみ、頭部カメラの向きが変わって見え率がずれる）
        cube_v = set()
        for name in [g.PER_SIZE and "cube_50mm", self.cube]:
            a = m.joint(name + "_free").dofadr[0]; cube_v |= set(range(a, a + 6))
        self.sim.gc_v = [v for v in range(m.nv) if v not in cube_v]
        key = m.key("stand"); n = m.nq - 14
        for j in range(m.njnt):
            nm = m.joint(j).name
            if nm.startswith("waist") or nm.startswith("left_"):
                a = m.jnt_qposadr[j]; d.qpos[a] = key.qpos[a]; d.qvel[m.jnt_dofadr[j]] = 0
        for j, v in LEFT_ARM.items():                  # 左腕は机に触れない位置へ（机に手が載ると腰が押されてカメラが傾く）
            a = m.joint(j).qposadr[0]; d.qpos[a] = v
            for i in range(m.nu):
                if m.actuator(i).name == j:
                    d.ctrl[i] = v
        mujoco.mj_forward(m, d)
        a = m.joint(self.cube + "_free").qposadr[0]
        h = np.radians(yaw_deg) / 2
        d.qpos[a:a + 7] = [cx + dxy[0], Y0 + dxy[1], bs.TABLE_TOP_Z + size / 2 + 0.001, np.cos(h), 0, 0, np.sin(h)]
        dv = m.joint(self.cube + "_free").dofadr[0]; d.qvel[dv:dv + 6] = 0
        mujoco.mj_forward(m, d)
        self.sim.step(0.3)
        self.c0 = d.body(self.cube).xpos.copy()          # 真の位置（評価用）
        self.board = m.geom("board").id
        self.table = m.geom("table").id
        self.cube_body = m.body(self.cube).id
        self.board_hits = set()
        self._orig_step = self.sim.mj_step
        self.sim.mj_step = self._step

    # --- 共通 ---
    def _step(self):
        self._orig_step()
        d, m = self.d, self.m
        for c in d.contact[:d.ncon]:
            if self.board in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.board else c.geom1
                self.board_hits.add(m.body(m.geom_bodyid[other]).name.replace("right_", ""))

    def grasp_point(self):
        w = self.d.body("right_wrist_yaw_link")
        return w.xpos + w.xmat.reshape(3, 3) @ self.sim.grasp_pt

    def move_linear(self, p_to, seconds, n=15):
        """挟む点を直線で動かす（途中点ごとにIK、関節目標を順に補間）。"""
        sim, m, d = self.sim, self.m, self.d
        p0 = self.grasp_point()
        save = d.qpos.copy(); qs, errs = [], []
        for k in range(1, n + 1):
            q, e = sim.ik(p0 + (np.asarray(p_to) - p0) * k / n); qs.append(q); errs.append(e)
            for a, v in zip(sim.arm_q, q):
                d.qpos[a] = v                          # 次のIKの初期値に（一時的に）
        d.qpos[:] = save; mujoco.mj_forward(m, d)
        q_prev = np.array([d.ctrl[sim.act[j]] for j in g.ARM])
        steps = max(1, int(seconds / m.opt.timestep / n))
        for q in qs:
            for i in range(steps):
                s = (i + 1) / steps
                for k, j in enumerate(g.ARM):
                    d.ctrl[sim.act[j]] = q_prev[k] + s * (q[k] - q_prev[k])
                sim.mj_step()
            q_prev = q
        return max(errs)

    def set_wrist_yaw(self, yaw):
        """手首の向きを切り替える。天板の下では回さず、手前（天板の外）へ引いてから回す。"""
        global YAW_DEG
        if abs(yaw - YAW_DEG) < 1e-9:
            return
        p = self.grasp_point()
        if p[0] > self.front_x - 0.05:                         # 天板の下にいるなら手前へ水平に引く
            self.move_linear(np.array([self.front_x - 0.10, p[1], p[2]]), 1.0)
        YAW_DEG = yaw
        p = self.grasp_point()
        self.sim.move_arm(np.array([p[0], p[1], max(p[2], self.c0[2] + PASS_H)]), 1.0)

    def approach(self, target_xy):
        """天板の手前で、挟む点を target の高さ＋3.5cm に置く（届く帯の中）。"""
        z = self.c0[2] + PASS_H
        p = self.grasp_point()
        if p[0] > self.front_x - 0.05:
            # 天板の下にいるときは、関節空間で動かすと手が物体を払うので、差し込むときと同じ高さにしてから手前へ直線で引く
            # 触った後は挟む点が差し込む高さより上にあり、そこから下げて引くと中指の先が部品の上面を引きずって
            # 部品を動かした（4回触ると数mm・数度）。天板の下では下げず、今の高さ以上のまま手前へ引く
            zr = max(p[2], z)
            self.move_linear(np.array([p[0], p[1], zr]), 0.5)
            self.move_linear(np.array([self.front_x - 0.10, p[1], zr]), 1.0)
        return self.sim.move_arm(np.array([self.front_x - 0.10, target_xy[1], z]), 2.0)

    def _middle_tip(self):
        """中指の一番低い点（世界座標）。"""
        m, d = self.m, self.d
        bid = m.body("right_hand_middle_1_link").id
        gid = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and m.geom_contype[i]][0]
        mid = m.geom_dataid[gid]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
        V = m.mesh_vert[a:a + n] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
        return V[np.argmin(V[:, 2])]

    def _thumb_clearance(self):
        """親指の先と、机・物体との最短距離[m]（5cm以上は5cm）。"""
        m, d = self.m, self.d
        bid = m.body("right_hand_thumb_2_link").id
        tg = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid and m.geom_contype[i]]
        out = {}
        for name, other in [("table", self.table), ("cube", m.geom(self.cube).id)]:
            out[name] = min(mujoco.mj_geomDistance(m, d, t, other, 0.05, None) for t in tg)
        return out

    # --- 触る ---
    def touch(self, target_xy, approach=True):
        """中指の先を target の真上から下ろし、0.1N を超えたら止める。
        戻り値：当たり（物体に触れた）か、接触相手、物体の移動[mm]、向きの変化[deg]、親指の最短距離[mm]"""
        m, d = self.m, self.d
        p_start = d.body(self.cube).xpos.copy()
        self.set_wrist_yaw(TOUCH_YAW)
        if approach:
            self.approach(target_xy)
        # 親指は中指より約5mm低いので、触る間だけ上へ退避させる（中指の形は掴むときと同じ）
        q = g.hand_q(g.PER_SIZE[self.cube]["s0"])
        q["right_hand_thumb_1_joint"], q["right_hand_thumb_2_joint"] = THUMB_UP
        if MIDDLE_TOUCH is not None:                   # 中指を曲げて指先を下に向ける（指の腹が先に当たらないように）
            q["right_hand_middle_0_joint"], q["right_hand_middle_1_joint"] = MIDDLE_TOUCH
        self.sim.set_hand(q); self.sim.step(0.3)
        off = self._middle_tip() - self.grasp_point()
        z_tab = bs.TABLE_TOP_Z
        top = self.c0[2] + self.size / 2
        start_z = top + TOUCH_START                                 # 指先を上面の少し上へ
        tgt = np.array([target_xy[0], target_xy[1], start_z])
        self.move_linear(tgt - off, 1.5)
        q0 = d.body(self.cube).xquat.copy(); p0 = d.body(self.cube).xpos.copy()
        mb = m.body("right_hand_middle_1_link").id
        hit, clear, loc = None, {"table": 1.0, "cube": 1.0}, None
        hit_z, hit_geom = None, None
        z = start_z
        while z > z_tab + TOUCH_FLOOR and hit is None:
            z -= 0.001
            self.move_linear(np.array([tgt[0], tgt[1], z]) - off, 0.02, n=1)
            for k in range(d.ncon):
                c = d.contact[k]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if mb in (b1, b2):
                    f = np.zeros(6); mujoco.mj_contactForce(m, d, k, f)
                    if f[0] > TOUCH_F:
                        hit = m.body(b2 if b1 == mb else b1).name
                        R = d.xmat[mb].reshape(3, 3); loc = R.T @ (c.pos - d.xpos[mb])
                        hit_z = float(c.pos[2]); hit_geom = m.geom(c.geom2 if b1 == mb else c.geom1).name; break
            tc = self._thumb_clearance()
            clear = {k: min(clear[k], tc[k]) for k in clear}
        tip = self._middle_tip()
        dq = abs(2 * np.degrees(np.arccos(min(1.0, abs(np.dot(q0, d.body(self.cube).xquat))))))
        on_cube = hit == self.cube
        dist_tip = float(np.linalg.norm(loc - TIP_CENTER)) if loc is not None else None
        obs = 0 if not on_cube else (1 if dist_tip <= TIP_RADIUS else 2)   # 0外れ／1指先の中央で当たり／2それ以外で当たり
        if on_cube and hit_geom is not None and hit_geom.endswith("_flange"):
            obs = 3                                                   # 3フランジに当たり（本体より低い）
        res = dict(hit=on_cube, obs=obs, partner=hit, tip_dist_mm=None if dist_tip is None else dist_tip * 1000,
                   loc_mm=None if loc is None else [float(v * 1000) for v in loc],
                   moved_mm=float(np.linalg.norm(d.body(self.cube).xpos[:2] - p0[:2]) * 1000),
                   yaw_deg=float(dq), tip_z_mm=float((tip[2] - z_tab) * 1000),
                   thumb_table_mm=float(clear["table"] * 1000), thumb_cube_mm=float(clear["cube"] * 1000),
                   hit_z_mm=None if hit_z is None else (hit_z - z_tab) * 1000, hit_geom=hit_geom)
        # 指を2cm上に戻し、親指を掴む形に戻す（次の動作のため）
        self.move_linear(np.array([tgt[0], tgt[1], start_z]) - off, 0.6, n=12)   # 3点で戻すと関節補間の曲がりで指が部品を約1mm引きずった
        self.sim.set_hand(g.hand_q(g.PER_SIZE[self.cube]["s0"])); self.sim.step(0.3)
        res["moved_total_mm"] = float(np.linalg.norm(d.body(self.cube).xpos[:2] - p_start[:2]) * 1000)   # 近づく・戻る動作も含めた移動
        return res

    # --- 掴む ---
    def grasp(self, target_xyz, squeeze=None):
        """挟む点を target に合わせて掴み、引き出して持ち上げる。成功＝2秒保持後に初期位置より4cm以上高い。"""
        if squeeze is not None:
            g.PER_SIZE[self.cube]["squeeze"] = squeeze
        t = np.asarray(target_xyz, dtype=float)
        self.set_wrist_yaw(GRASP_YAW)
        errs = [self.approach(t[:2])]
        errs.append(self.move_linear(t + [0, 0, PASS_H], 1.5))   # 水平に差し込む
        errs.append(self.move_linear(t, 1.0))                    # 真下に下ろす
        hold = close_each(self.sim, 1.0); self.sim.step(0.3)
        self.move_linear(t + [0, 0, 0.01], 0.5)                          # 1cm浮かせる
        self.move_linear(np.array([self.front_x - 0.10, t[1], t[2] + 0.01]), 1.5)  # 手前へ引き出す
        self.move_linear(np.array([self.front_x - 0.10, t[1], t[2] + 0.06]), 1.0)  # 持ち上げる
        self.sim.step(2.0)                                                  # 2秒保持
        rise = self.d.body(self.cube).xpos[2] - self.c0[2]
        return dict(success=bool(rise > 0.04), rise_cm=float(rise * 100), closed=hold is not None,
                    board=sorted(self.board_hits), ik_err_mm=float(max(errs) * 1000))
