"""図3（配置64での触る点の選び方の違い）を作る：提案手法（voi）と不確かさ最大（unc）の試行から
「① 触る」「② 挟む」「③ 引き出して持ち上げる」の3コマを抜き出して並べる。

使い方（リポジトリの直下で。2つの試行を動かすので数分かかる）:
  uv run python tools/fig_layout64.py docs/figures/fig_layout64.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import view_trial as V

LAYOUT = 64
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
font = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s, index=0)
BLUE, RED = (20, 80, 170), (180, 30, 30)


def aim(self, cam):                                   # 部品のまわりに寄った視点（動画より近い）
    c = self.s.c0
    cam.lookat[:] = [c[0] + 0.005, c[1], c[2] + 0.025]
    cam.distance, cam.azimuth, cam.elevation = 0.24, 60.0, -16.0


def record(rule):
    """1回触って掴む試行を動かし、各コマの画像（文字なし）・説明文・中指の先の高さ・部品の高さを返す。"""
    rec = []
    def caption(self, img, head=None):
        s = self.s
        rec.append((img.copy(), self.text, float(s._middle_tip()[2]), float(s.d.body(s.cube).xpos[2])))
        return img
    V.Show._aim, V.Show.caption = aim, caption
    tmp = f"/tmp/_layout{LAYOUT}_{rule}.mp4"
    ok = V.run(LAYOUT, rule, 1, video=tmp)
    os.remove(tmp)
    return rec, ok


def pick(rec):
    """① 触る：触る間で中指の先が最も低いコマ。② 挟む：掴む間で中指の先が最も低いコマの0.6秒後。
    ③ 引き出す：同じく1.8秒後（手が天板の前端あたりまで戻ったところ）。"""
    tz = [(z, i) for i, (_, t, z, _) in enumerate(rec) if "touch" in t.split("\n")[-1]]
    gz = [(z, i) for i, (_, t, z, _) in enumerate(rec) if t.split("\n")[-1].startswith("grasp")]
    i1 = min(tz)[1]
    lo, last = min(gz)[1], gz[-1][1]
    i2, i3 = min(lo + int(0.6 * V.FPS), last), min(lo + int(1.8 * V.FPS), last)
    return [rec[i1][0], rec[i2][0], rec[i3][0]]


def main(out):
    rows = []
    for rule in ("voi", "unc"):
        rec, ok = record(rule)
        rows.append((pick(rec), ok))
    cw, ch = 640, 363                                   # 960×544 を 2/3 に
    W, H = 60 + 3 * cw + 20, 50 + 2 * (ch + 10) + 40
    fig = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(fig)
    for k, lab in enumerate(["① 触る（中指で上から）", "② 挟む", "③ 引き出して持ち上げる"]):
        dr.text((70 + k * cw, 14), lab, font=font(28, True), fill=(0, 0, 0))
    for r, ((frames, ok), name, col) in enumerate(zip(rows, ["提案手法", "不確かさ最大"], [BLUE, RED])):
        y = 50 + r * (ch + 10)
        for k, f in enumerate(frames):
            fig.paste(Image.fromarray(f).resize((cw, ch)), (60 + k * cw, y))
        msg = "掴めた" if ok else "掴み損ねた（部品が落ちる）"
        tw = dr.textlength(msg, font=font(28, True))
        x1 = 60 + 3 * cw - 10
        dr.rectangle([x1 - tw - 16, y + 10, x1, y + 54], fill="white")
        dr.text((x1 - tw - 8, y + 14), msg, font=font(28, True), fill=col)
        lab = Image.new("RGB", (ch, 50), "white"); ImageDraw.Draw(lab).text(
            (ch / 2 - 14 * len(name), 6), name, font=font(28, True), fill=(0, 0, 0))
        fig.paste(lab.rotate(90, expand=True), (5, y))
    dr.text((60, H - 34), "赤：触る点　青：掴む点　黄：形の候補ごとの本体の手前の縁（200個）　　配置64、見えている長さ10mm",
            font=font(22), fill=(80, 80, 80))
    fig.save(out); print("保存:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig_layout64.png")
