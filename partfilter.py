"""フランジ付き部品の推定と判断（粒子で平均した成功確率、触る観測の尤度、触る点の選び方）。

粒子の状態：(手前の面のx, 中心のy, yaw, フランジの長さ fl, 本体の長さ L)。
部品の軸に沿った座標 a（手前の面から奥へ[m]）で考える。本体は a∈[fl, fl+L]。
"""
import numpy as np

# 触る観測（予備実験：本体の手前の縁を基準にした奥行き u、奥の縁を基準にした v）
#   u < −11mm：フランジ（a>=0 のとき）／ −11〜−3mm：指の腹で本体 ／ −3mm〜v=+2mm：指先の中央で本体
#   v = +2〜+12mm：指の腹で本体 ／ それより奥：外れ
MISS, CENTER, OTHER, FLANGE = 0, 1, 2, 3
B_PAD_FRONT, B_TIP_FRONT, B_TIP_BACK, B_PAD_BACK = -0.011, -0.003, 0.002, 0.012
BLUR = 0.001            # 境界のぼかし[m]（位置の誤差と指の形のばらつき）
EPS = 0.02              # どの観測にも残す確率（まれな外れ値で粒子が全滅しないように）

# 掴める範囲（予備実験：本体の手前の縁から3mm以上、奥の縁から1.5mm以上内側。横は±4mmで8回中8回）
IN_FRONT, IN_BACK = 0.0035, 0.0025   # 測定の刻み1.5mmと向き・横のずれを見込んだ余裕
LAT_OK = 0.005          # 挟む点の横のずれの許容[m]
W = 0.020               # 部品の幅[m]
LAT_TOUCH = 0.003       # 指先が部品の横の縁からこの距離までは当たる[m]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z / BLUR))


def axial(parts, x, y):
    """点 (x,y) の、各粒子の部品軸に沿った座標 a と横のずれ b。"""
    dx, dy = x - parts[:, 0], y - parts[:, 1]
    c, s = np.cos(parts[:, 2]), np.sin(parts[:, 2])
    return c * dx + s * dy, -s * dx + c * dy


def touch_lik(parts, x, y, obs):
    """観測 obs が得られる確率を粒子ごとに返す。"""
    a, b = axial(parts, x, y)
    u = a - parts[:, 3]; v = a - parts[:, 3] - parts[:, 4]
    inside = _sig(W / 2 + LAT_TOUCH - np.abs(b))              # 横に外れていれば何にも当たらない
    body_pad_f = _sig(u - B_PAD_FRONT) * (1 - _sig(u - B_TIP_FRONT))
    body_tip = _sig(u - B_TIP_FRONT) * (1 - _sig(v - B_TIP_BACK))
    body_pad_b = _sig(v - B_TIP_BACK) * (1 - _sig(v - B_PAD_BACK))
    rest = 1 - body_pad_f - body_tip - body_pad_b
    on_fl = _sig(a) * (1 - _sig(u - B_PAD_FRONT))              # フランジの上（本体の角に指が届く手前まで）
    p = {CENTER: body_tip * inside, OTHER: (body_pad_f + body_pad_b) * inside, FLANGE: np.minimum(on_fl, rest) * inside}
    p[MISS] = np.clip(1 - p[CENTER] - p[OTHER] - p[FLANGE], 0, 1)
    return EPS / 4 + (1 - EPS) * p[obs]


def grasp_prob(parts, x, y):
    """挟む点 (x,y) で掴める確率（粒子の平均）。予測器ができるまでの幾何の代わり。"""
    a, b = axial(parts, x, y)
    ok = (a >= parts[:, 3] + IN_FRONT) & (a <= parts[:, 3] + parts[:, 4] - IN_BACK) & (np.abs(b) <= LAT_OK)
    return float(ok.mean())


def best_grasp(parts, cands):
    """候補の挟む点のうち、成功確率が最大のもの。"""
    ps = [grasp_prob(parts, x, y) for x, y in cands]
    i = int(np.argmax(ps)); return cands[i], ps[i]


def axis_points(parts, step=0.001, lo=0.0, hi=0.055):
    """推定の平均の軸に沿った候補点（手前の面から lo〜hi）。"""
    xf, y, yaw = parts[:, 0].mean(), parts[:, 1].mean(), np.arctan2(np.sin(parts[:, 2]).mean(), np.cos(parts[:, 2]).mean())
    a = np.arange(lo, hi + 1e-9, step)
    return [(xf + np.cos(yaw) * t, y + np.sin(yaw) * t) for t in a]


def update(parts, x, y, obs, rng):
    """触った結果で重みを付けて再標本化する。"""
    w = touch_lik(parts, x, y, obs); w = w / w.sum()
    idx = rng.choice(len(parts), len(parts), p=w)
    out = parts[idx] + rng.normal(0, [0.0003, 0.0003, np.radians(0.5), 0.0003, 0.0003], parts.shape)
    return out


def choose_touch(parts, grasp_cands, touch_cands):
    """触った後の最良の成功確率の期待値が最大になる触る点（1手先読み）。"""
    best = (-1.0, None)
    for x, y in touch_cands:
        ev = 0.0
        for o in (MISS, CENTER, OTHER, FLANGE):
            lk = touch_lik(parts, x, y, o); po = lk.mean()
            if po < 1e-6:
                continue
            w = lk / lk.sum()
            ps = []
            for gx, gy in grasp_cands:
                aa, bb = axial(parts, gx, gy)
                ok = (aa >= parts[:, 3] + IN_FRONT) & (aa <= parts[:, 3] + parts[:, 4] - IN_BACK) & (np.abs(bb) <= LAT_OK)
                ps.append(float((w * ok).sum()))
            ev += po * max(ps)
        if ev > best[0]:
            best = (ev, (x, y))
    return best[1], best[0]


# --- 学習した予測器（形の候補ごとの成功確率）を使う版 ---
_PRED = None
_BOARD_FRONT = None


def use_predictor(path, board_front):
    """予測器を読み込み、以後の成功確率はそれで出す。board_front は天板の前端のx（奥行きの入力に使う）。"""
    import pickle
    global _PRED, _BOARD_FRONT
    _PRED = pickle.load(open(path, "rb"))["model"]; _BOARD_FRONT = board_front


def prob_matrix(parts, cands):
    """粒子×挟む点の成功確率の行列。予測器がなければ幾何の規則。"""
    cands = np.asarray(cands)
    dx = cands[None, :, 0] - parts[:, 0:1]; dy = cands[None, :, 1] - parts[:, 1:2]
    c, s = np.cos(parts[:, 2:3]), np.sin(parts[:, 2:3])
    a = c * dx + s * dy; b = -s * dx + c * dy
    fl, L = parts[:, 3:4], parts[:, 4:5]
    if _PRED is None:
        return ((a >= fl + IN_FRONT) & (a <= fl + L - IN_BACK) & (np.abs(b) <= LAT_OK)).astype(float)
    n, g = a.shape
    X = np.column_stack([((a - fl) * 1000).ravel(), ((fl + L - a) * 1000).ravel(), (b * 1000).ravel(),
                         np.repeat(np.degrees(parts[:, 2]), g), np.repeat(L[:, 0] * 1000, g),
                         np.tile((cands[:, 0] - _BOARD_FRONT) * 1000, n)])
    return _PRED.predict_proba(X)[:, 1].reshape(n, g)


def best_grasp_m(parts, cands):
    P = prob_matrix(parts, cands).mean(0); i = int(np.argmax(P))
    return tuple(cands[i]), float(P[i])


def choose_touch_m(parts, grasp_cands, touch_cands):
    """触った後の最良の成功確率の期待値が最大になる触る点（1手先読み、予測器の行列を使い回す）。"""
    Pm = prob_matrix(parts, grasp_cands)
    best = (-1.0, None)
    for x, y in touch_cands:
        ev = 0.0
        for o in (MISS, CENTER, OTHER, FLANGE):
            lk = touch_lik(parts, x, y, o); s = lk.sum()
            if s / len(parts) < 1e-6:
                continue
            ev += (s / len(parts)) * float(((lk / s) @ Pm).max())
        if ev > best[0]:
            best = (ev, (x, y))
    return best[1], best[0]
