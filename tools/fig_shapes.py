"""図1（形の族の例）を作る：本体の長さが違う3つの部品を斜め上から描き、同じ配置を頭部カメラから見た画像を並べる。

使い方（リポジトリの直下で）:
  uv run python tools/fig_shapes.py docs/figures/fig_shapes.png
3つとも見えている長さ 10mm（フランジの手前だけ見える）。フランジ＋本体の長さは 32mm（a）、36mm（b）、40mm（c）。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import shelf as S
import mujoco
import build_scene as bs
from PIL import Image, ImageDraw, ImageFont

PARTS = [(0.012, 0.020), (0.020, 0.016), (0.028, 0.012)]   # (フランジの長さ, 本体の長さ)
VISIBLE = 0.010
W_OV, H_OV = 800, 600
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s, index=0)


def project(cam, fovy, P):
    """自由カメラ（lookat・距離・方位・仰角）で、世界座標の点を画素 (列, 行) に移す。"""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = np.array(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    v = np.asarray(P) - pos
    f = H_OV / 2 / np.tan(np.radians(fovy) / 2)
    z = v @ fwd
    return W_OV / 2 + f * (v @ right) / z, H_OV / 2 - f * (v @ up) / z


def render_part(fl, lb):
    s = S.ShelfSim(0.020, visible=VISIBLE, part=(fl, lb))
    m, d = s.m, s.d
    # 頭部カメラ（天板は不透明のまま）
    hc = mujoco.Renderer(m, 480, 640); hc.update_scene(d, "head_cam"); head = hc.render(); hc.close()
    # 斜め上からの全体図（天板を半透明、影なし）
    m.geom_rgba[m.geom("board").id] = [0, 0, 0, 0]          # 天板は描かず、隠れる範囲を網掛けで示す
    m.light_castshadow[:] = 0
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.vis.global_.offwidth, m.vis.global_.offheight = W_OV, H_OV
    rd = mujoco.Renderer(m, H_OV, W_OV)
    cam = mujoco.MjvCamera()
    c0 = s.c0
    xf = S.X_FRONT                                             # 部品の手前の面
    cam.lookat[:] = [xf + 0.022, c0[1], bs.TABLE_TOP_Z + 0.012]
    cam.distance, cam.azimuth, cam.elevation = 0.13, 30.0, -24.0
    rd.update_scene(d, cam)
    # 右腕・ロボット本体は描かない（部品のまわりだけ）
    for i in range(rd.scene.ngeom):
        g = rd.scene.geoms[i]
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[g.objid] not in (0, m.body(s.cube).id):
            g.rgba[3] = 0
    img = rd.render()
    z = bs.TABLE_TOP_Z; y = c0[1] - 0.016
    ann = {k: project(cam, m.vis.global_.fovy, [v, y, z]) for k, v in [("f0", xf), ("f1", xf + fl), ("b1", xf + fl + lb)]}
    ann["vis0"] = project(cam, m.vis.global_.fovy, [xf + VISIBLE, c0[1] - 0.03, z + bs.FLANGE_H])
    ann["vis1"] = project(cam, m.vis.global_.fovy, [xf + VISIBLE, c0[1] + 0.03, z + bs.FLANGE_H])
    xv = xf + VISIBLE; zt = z + 0.024; y0_, y1_ = c0[1] - 0.03, c0[1] + 0.03
    ann["shade_top"] = [project(cam, m.vis.global_.fovy, q) for q in
                        ([xv, y0_, zt], [xv + 0.08, y0_, zt], [xv + 0.08, y1_, zt], [xv, y1_, zt])]
    ann["shade_front"] = [project(cam, m.vis.global_.fovy, q) for q in
                          ([xv, y0_, z], [xv, y0_, zt], [xv, y1_, zt], [xv, y1_, z])]
    rd.close()
    import perception as P
    hu, hv = P.HeadCam(m, d).project([c0])[0]
    ann["head"] = (float(hu), float(hv))
    return img, head, ann


def main(out):
    W, H = 1720, 750
    fig = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(fig)
    for k, (fl, lb) in enumerate(PARTS):
        img, head, a = render_part(fl, lb)
        im = Image.fromarray(img).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); do = ImageDraw.Draw(ov)
        do.polygon(a["shade_top"], fill=(60, 60, 75, 95)); do.polygon(a["shade_front"], fill=(60, 60, 75, 70))
        im = Image.alpha_composite(im, ov).convert("RGB"); di = ImageDraw.Draw(im)
        # 天板に隠れる境界（点線）
        (x0, y0), (x1, y1) = a["vis0"], a["vis1"]
        for t in np.linspace(0, 1, 40)[::2]:
            p = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t); q = (x0 + (x1 - x0) * (t + 0.025), y0 + (y1 - y0) * (t + 0.025))
            di.line([p, q], fill=(30, 30, 60), width=3)
        # 寸法線
        def dim(p, q, lab, off):
            p = (p[0], p[1] + off); q = (q[0], q[1] + off)
            di.line([p, q], fill=(0, 0, 0), width=2)
            for r in (p, q):
                di.line([(r[0], r[1] - 7), (r[0], r[1] + 7)], fill=(0, 0, 0), width=2)
            di.text(((p[0] + q[0]) / 2 - 30, (p[1] + q[1]) / 2 + 4), lab, font=font(22), fill=(0, 0, 0))
        dim(a["f0"], a["f1"], f"ℓ_f={fl * 1000:.0f}", 22)
        dim(a["f1"], a["b1"], f"ℓ_b={lb * 1000:.0f}", 22)
        im = im.crop((0, 60, 800, 574)).resize((560, 360))
        fig.paste(im, (20 + 560 * k, 60))
        dr.text((30 + 560 * k, 18), f"({'abc'[k]}) 本体の長さ {lb * 1000:.0f}mm（フランジ {fl * 1000:.0f}mm）",
                font=font(24, True), fill=(0, 0, 0))
        # 頭部カメラ：部品のまわりを切り出して拡大
        h = Image.fromarray(head)
        pu, pv = a["head"]                                      # 部品の中心の画素（640×480 の画像）
        x0 = int(np.clip(pu - 107, 0, 640 - 214)); y0 = int(np.clip(pv - 95, 0, 480 - 160))
        crop = h.crop((x0, y0, x0 + 214, y0 + 160)).resize((320, 240))
        fig.paste(crop, (140 + 560 * k, 470))
    dr.text((30, 436), "↓ 頭部カメラから見ると（部品のまわりを拡大）：3つとも同じに見える", font=font(24, True), fill=(0, 0, 0))
    dr.text((24, 718), "点線より奥（網掛け）は天板に隠れてカメラから見えない（見えている長さ 10mm）。寸法の単位は mm。",
            font=font(21), fill=(80, 80, 80))
    fig.save(out); print("保存:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig_shapes.png")
