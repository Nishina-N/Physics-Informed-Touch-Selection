"""頭部カメラから物体の大きさと位置の初期分布（粒子）を作る。

- 観測：物体のマスク（シムでは正解の領域分割）と深度画像（D435iの雑音モデル σ(z)=z²·0.08/(f·b)）。
- 大きさ：上面と周囲10cmの机面の深度差（高さ）。立方体なので幅＝高さとして使う。
- 位置：粒子ごとに「その位置なら見えるはずの領域」を作り、観測マスクと照合して重みを付ける。
  見えるはずの領域 ＝ 立方体の8頂点を投影した凸包 − 手前にある天板・ロボットの画素（観測画像から分かる）。
  見えている部分の重心は見えている側に偏るので使わない。
"""
import numpy as np
import mujoco
import cv2

W, H = 640, 480
BASELINE = 0.05          # D435i の基線長[m]
N_PRIOR = 3000           # 事前分布から引く候補の数
N_PART = 200             # 粒子の数
TEMP = 0.02              # 照合の重み exp(−不一致率/TEMP) の温度（較正で決める）


class HeadCam:
    def __init__(self, m, d):
        self.m, self.d = m, d
        cid = m.camera("head_cam").id
        self.f = (H / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
        self.R = d.cam_xmat[cid].reshape(3, 3).copy()   # 列＝カメラのx,y,z軸（世界）
        self.p = d.cam_xpos[cid].copy()

    def render(self, rng):
        m, d = self.m, self.d
        rd = mujoco.Renderer(m, H, W); rd.enable_depth_rendering(); rd.update_scene(d, "head_cam"); dep = rd.render(); rd.close()
        rs = mujoco.Renderer(m, H, W); rs.enable_segmentation_rendering(); rs.update_scene(d, "head_cam"); ids = rs.render(); rs.close()
        dep = dep + rng.normal(0, 1, dep.shape) * dep ** 2 * 0.08 / (self.f * BASELINE)
        return dep, ids

    def to_world(self, dep):
        v, u = np.mgrid[0:H, 0:W]
        x = (u - W / 2 + 0.5) / self.f * dep; y = -(v - H / 2 + 0.5) / self.f * dep
        return np.stack([x, y, -dep], -1) @ self.R.T + self.p

    def project(self, P):
        c = (np.asarray(P) - self.p) @ self.R            # カメラ座標
        z = -c[:, 2]
        return np.stack([W / 2 + self.f * c[:, 0] / z - 0.5, H / 2 - self.f * c[:, 1] / z - 0.5], -1)


MASK_NOISE = True        # 領域分割の境界の誤り（実機の領域分割モデルを想定）


def noisy_mask(mask, rng):
    """境界を±1画素ずらし、境界の画素を30%の確率で反転する。"""
    k = np.ones((3, 3), np.uint8); m8 = mask.astype(np.uint8)
    r = rng.random()
    if r < 1 / 3:
        m8 = cv2.erode(m8, k)
    elif r < 2 / 3:
        m8 = cv2.dilate(m8, k)
    edge = (cv2.dilate(m8, k) - cv2.erode(m8, k)).astype(bool)
    flip = edge & (rng.random(mask.shape) < 0.3)
    return m8.astype(bool) ^ flip


def observe(m, d, cube, rng):
    """頭部カメラの観測：物体マスク、手前の遮蔽物（天板・ロボット）のマスク、世界座標の点群。"""
    cam = HeadCam(m, d)
    dep, ids = cam.render(rng)
    geom = ids[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    gids = [m.geom(cube).id]
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, cube + "_flange") >= 0:   # フランジ付き部品は両方の形状
        gids.append(m.geom(cube + "_flange").id)
    cube_mask = geom & np.isin(ids[..., 0], gids)
    true_mask = cube_mask
    if MASK_NOISE:
        cube_mask = noisy_mask(cube_mask, rng)
    table_mask = geom & (ids[..., 0] == m.geom("table").id)
    occl = geom & ~true_mask & ~table_mask               # 天板・ロボット・他の物体（物体より手前にあり得るもの）
    return cam, dep, cube_mask, table_mask, occl


def estimate_size(cam, dep, cube_mask, table_mask, floor=0.001):
    P = cam.to_world(dep)
    c = P[cube_mask]
    if len(c) < 5:
        return None
    cxy = np.median(c[:, :2], 0)
    near = table_mask & (np.linalg.norm(P[..., :2] - cxy, axis=-1) < 0.10)
    ztab = np.median(P[near][:, 2])
    # 上面の高さ：見えている点の上位5%の中央値（奥が隠れると上面の見える割合が減るので、上位の点だけを使う）
    c = c[(c[:, 2] > ztab + 0.002) & (c[:, 2] < ztab + 0.05)]   # 境界の誤りで混じる天板や机の点を除く（物体は5cm以下）
    if len(c) < 5:
        return None
    top = c[c[:, 2] >= np.percentile(c[:, 2], 95)]
    return float(np.median(top[:, 2]) - ztab), float(ztab)


def box_corners(x, y, yaw, s, ztab):
    h = s / 2; c, sn = np.cos(yaw), np.sin(yaw)
    pts = []
    for dx in (-h, h):
        for dy in (-h, h):
            for z in (ztab, ztab + s):
                pts.append([x + c * dx - sn * dy, y + sn * dx + c * dy, z])
    return np.array(pts)


def init_particles(cam, cube_mask, occl, size, ztab, rng, prior_xy, prior_sd=(0.03, 0.015)):
    """隠れを考慮した照合で粒子の初期分布を作る。prior_xy は大まかな事前の中心（物体があり得る範囲の中心）。"""
    ys, xs = np.nonzero(cube_mask)
    u0, u1, v0, v1 = xs.min() - 60, xs.max() + 60, ys.min() - 60, ys.max() + 60
    u0, v0 = max(u0, 0), max(v0, 0); u1, v1 = min(u1, W - 1), min(v1, H - 1)
    obs = cube_mask[v0:v1 + 1, u0:u1 + 1]; blk = occl[v0:v1 + 1, u0:u1 + 1]
    n_obs = obs.sum()
    cand = np.column_stack([rng.normal(prior_xy[0], prior_sd[0], N_PRIOR),
                            rng.normal(prior_xy[1], prior_sd[1], N_PRIOR),
                            rng.uniform(-np.pi / 4, np.pi / 4, N_PRIOR)])
    err = np.empty(N_PRIOR)
    buf = np.zeros_like(obs, dtype=np.uint8)
    for i, (x, y, yaw) in enumerate(cand):
        uv = cam.project(box_corners(x, y, yaw, size, ztab)) - [u0, v0]
        buf[:] = 0
        cv2.fillConvexPoly(buf, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        pred = buf.astype(bool) & ~blk
        err[i] = (pred ^ obs).sum() / max(n_obs, 1)       # 見えるはずの領域と観測のずれ（観測の画素数で割る）
    return resample(cand, err, TEMP, rng), (cand, err)


def box_corners_x(x0, x1, y, yaw, w, h, ztab, xf):
    """奥行き x0〜x1（手前の面 xf を基準に yaw 回転）、幅 w、高さ h の箱の8頂点。"""
    c, sn = np.cos(yaw), np.sin(yaw); pts = []
    for dx in (x0 - xf, x1 - xf):
        for dy in (-w / 2, w / 2):
            for z in (ztab, ztab + h):
                pts.append([xf + c * dx - sn * dy, y + sn * dx + c * dy, z])
    return np.array(pts)


FL_H = 0.004             # フランジの高さ（build_scene.FLANGE_H と同じ）
K3 = np.ones((3, 3), np.uint8)
YAW_MAX = np.radians(25)   # 部品の向きの事前範囲（棚に置かれた部品はおおむね正面向き。実際は±20°）


YAW_SD = np.radians(3.5)   # 深度から求めた向きの誤差（40配置で平均2.3°、90%点3.9°、最大5.1°）


def yaw_from_depth(cam, dep, mask, ztab, w=0.020):
    """フランジの手前の縁を深度で直線に当てはめて部品の向きを出す。
    領域の形だけでは向きがほとんど決まらなかった（推定が事前分布の中央0°に寄り、真の向きとの相関が−0.92）。
    手前の縁の向き（dx/dy = −tan yaw）から yaw を得る。点が足りなければ None。"""
    Pw = cam.to_world(dep)[mask]
    Pw = Pw[(Pw[:, 2] > ztab + 0.0015) & (Pw[:, 2] < ztab + 0.007)]   # フランジの上面と手前の面（本体の点を混ぜない）
    if len(Pw) < 20:
        return None
    y0 = np.median(Pw[:, 1]); edges = np.arange(y0 - 0.007, y0 + 0.007 + 1e-9, 0.002)   # 左右の端は領域の誤りが多いので中央14mm
    ys, xs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        q = Pw[(Pw[:, 1] >= a) & (Pw[:, 1] < b)]
        if len(q) >= 3:
            ys.append((a + b) / 2); xs.append(np.percentile(q[:, 0], 5))
    if len(ys) < 3:
        return None
    t = np.polyfit(ys, xs, 1)[0]
    return float(-np.arctan(t))


def init_particles_part(cam, mask, occl, w, ztab, rng, prior_xy, fl_range, L_range, prior_sd=(0.03, 0.015), yaw0=None):
    """フランジ付き部品の粒子。状態＝(手前の面のx, 中心のy, yaw, フランジの長さ, 本体の長さ)。
    見えるはずの領域＝フランジと本体それぞれの凸包の和 − 手前の遮蔽物。本体が見えないことも照合で効く。"""
    ys, xs = np.nonzero(mask)
    u0, u1, v0, v1 = xs.min() - 60, xs.max() + 60, ys.min() - 60, ys.max() + 60
    u0, v0 = max(u0, 0), max(v0, 0); u1, v1 = min(u1, W - 1), min(v1, H - 1)
    obs = mask[v0:v1 + 1, u0:u1 + 1]; blk = occl[v0:v1 + 1, u0:u1 + 1]
    n_obs = obs.sum()
    cand = np.column_stack([rng.normal(prior_xy[0], prior_sd[0], N_PRIOR),
                            rng.normal(prior_xy[1], prior_sd[1], N_PRIOR),
                            (rng.uniform(-YAW_MAX, YAW_MAX, N_PRIOR) if yaw0 is None
                             else rng.normal(yaw0, YAW_SD, N_PRIOR)),
                            rng.uniform(*fl_range, N_PRIOR),
                            rng.uniform(*L_range, N_PRIOR)])
    err = np.empty(N_PRIOR)
    buf = np.zeros_like(obs, dtype=np.uint8)
    for i, (xf, y, yaw, fl, L) in enumerate(cand):
        buf[:] = 0
        for x0, x1, h in ((xf, xf + fl, FL_H), (xf + fl, xf + fl + L, w)):
            uv = cam.project(box_corners_x(x0, x1, y, yaw, w, h, ztab, xf)) - [u0, v0]
            cv2.fillConvexPoly(buf, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        # 領域分割の境界は±1画素ずれ得るので、予測の境界1画素は照合に使わない（部品は幅20画素ほどと小さく、
        # 境界の誤りをそのまま数えると「本体が隠れている」側に偏った）
        p_in = cv2.erode(buf, K3).astype(bool) & ~blk
        p_out = cv2.dilate(buf, K3).astype(bool) & ~blk
        err[i] = ((p_in & ~obs).sum() + (obs & ~p_out).sum()) / max(n_obs, 1)
    w_ = np.exp(-(err - err.min()) / TEMP); w_ /= w_.sum()
    idx = rng.choice(len(cand), N_PART, p=w_)
    jit = rng.normal(0, [0.0005, 0.0005, np.radians(1), 0.0005, 0.0005], (N_PART, 5))
    parts = cand[idx] + jit
    parts[:, 3] = np.clip(parts[:, 3], *fl_range); parts[:, 4] = np.clip(parts[:, 4], *L_range)
    return parts, (cand, err)


BODY_Z = 0.010           # これより高い点は本体（フランジは4mm、深度の雑音は約1mm）
BODY_SD = 0.0015         # 見えた本体の手前の縁から決める fl のばらつき[m]


def refine_part_depth(parts, cam, dep, mask, ztab, board_front, board_z, fl_range, rng):
    """深度で本体が見えているかを調べ、フランジの長さ fl を引き直す。
    領域の形だけでは、本体の奥行きの位置を変えても見える形がほとんど変わらなかった（照合の誤差の差が境界の誤りより小さい）。
    本体が見えていれば：見えた本体の点の最も手前（部品の軸に沿って）＝本体の手前の縁。
    見えていなければ：高さ BODY_Z で天板の縁に隠れ始める位置より奥に本体の手前の縁がある。"""
    Pw = cam.to_world(dep)[mask]
    tall = Pw[(Pw[:, 2] > ztab + BODY_Z) & (Pw[:, 2] < ztab + 0.03)]   # 天板など上の混入点を除く
    out = parts.copy(); n = len(parts)
    c, s = np.cos(parts[:, 2]), np.sin(parts[:, 2])
    if len(tall) >= 5:
        a = c[:, None] * (tall[None, :, 0] - parts[:, 0:1]) + s[:, None] * (tall[None, :, 1] - parts[:, 1:2])
        a_min = np.percentile(a, 30, axis=1)          # 手前の面の点は真の縁の±1mm（30%点で約−0.3mm）
        # 手前の面の推定の誤差（数mm）を fl が吸収するので、事前の範囲で切らない（本体の縁の絶対位置を保つ）
        out[:, 3] = np.clip(a_min + rng.normal(0, BODY_SD, n), fl_range[0] - 0.008, fl_range[1] + 0.008)
        return out, True
    h = ztab + BODY_Z
    x_hide = cam.p[0] + (board_front - cam.p[0]) * (cam.p[2] - h) / (cam.p[2] - board_z)
    lo = np.clip(x_hide - parts[:, 0], fl_range[0], fl_range[1])
    out[:, 3] = rng.uniform(lo, fl_range[1])
    return out, False


def resample(cand, err, temp, rng):
    w = np.exp(-(err - err.min()) / temp); w /= w.sum()
    idx = rng.choice(len(cand), N_PART, p=w)
    return cand[idx] + rng.normal(0, [0.0005, 0.0005, np.radians(1)], (N_PART, 3))   # 同じ候補の重複をわずかにほぐす


def summarize(parts):
    yaw = parts[:, 2]
    return dict(mean=parts[:, :2].mean(0), sd=parts[:, :2].std(0),
                yaw_mean=float(np.degrees(yaw.mean())), yaw_sd=float(np.degrees(yaw.std())))
