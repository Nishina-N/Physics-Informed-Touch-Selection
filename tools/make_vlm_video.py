"""SmolVLM との組み合わせの比較動画（docs/videos/vlm_compare.mp4）を作る。

材料：vlm_pick.py で作った2本（同じ場面・同じ選択で、触らずに掴む／1回触ってから掴む）と、results/vlmcmp_*.jsonl の集計。
  uv run python vlm_pick.py --layout 0 --sizes <run_vlm_compare.sizes_for(0)> --answer 2 --touches 0 --video docs/videos/vlm_pick_t0.mp4
  uv run python vlm_pick.py ...（同じ）--touches 1 --video docs/videos/vlm_pick_t1.mp4
  uv run python tools/make_vlm_video.py [--answer-note "SmolVLM-500M の答え"]
--answer を手で与えて作った動画なら、そのことを字幕に出す（既定）。SmolVLM で答えさせた動画なら --answer-note を変える。
"""
import argparse, glob, json
import numpy as np, imageio.v2 as io
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--answer-note", default="（この動画では SmolVLM の答えの代わりに番号を手で与えた）")
a = ap.parse_args()
VID = "docs/videos/"
W, H, FPS = 1280, 720, 30
R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
f = lambda s, b=False: ImageFont.truetype(B if b else R, s, index=0)
BG = (250, 250, 248); FG = (25, 28, 30); MUTED = (95, 100, 105); ACC = (30, 95, 170); RED = (175, 45, 45)
out = io.get_writer(VID + "vlm_compare.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=16)


def emit(img, sec):
    x = np.array(img.convert("RGB"))
    for _ in range(int(sec * FPS)):
        out.append_data(x)


def card(title, lines=(), step=None):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if step:
        d.text((60, 34), step, font=f(22), fill=MUTED)
    d.text((60, 64), title, font=f(40, True), fill=FG)
    y = 140
    for ln in lines:
        s, col = (ln if isinstance(ln, tuple) else (ln, FG))
        d.text((70, y), s, font=f(28), fill=col); y += 48
    return im, d


# 1 課題と分担
im, d = card("SmolVLM ＋ 触る点の選択：指示文で選び、触って確かめてから掴む",
             ["棚の下に、色と隠れたサイズの違う部品が2つ。カメラにはどちらもフランジの手前しか見えない。",
              "① SmolVLM：指示文と頭部カメラの画像から、取る部品を番号で選ぶ（見える色で判断）",
              "② 提案手法：選んだ部品を1回触って隠れたサイズを確かめ、掴む点を選ぶ",
              ("比較：② を省き、カメラの推定だけで掴む（SmolVLM で選んでそのまま掴む）", RED),
              "", ("どちらも手の動き（経路）は決めた経由点を逆運動学で解くもの。違うのは触るかどうかだけ", MUTED)],
             step="SmolVLM との組み合わせ")
emit(im, 12)
# 2 SmolVLM の入力と答え（vlm_pick の最初の画面をそのまま使う）
A = [x for x in io.get_reader(VID + "vlm_pick_t1.mp4")]
hh = H - 50; im = Image.fromarray(A[0]).resize((int(hh * A[0].shape[1] / A[0].shape[0]), hh))
canvas = Image.new("RGB", (W, H), (0, 0, 0)); canvas.paste(im, ((W - im.width) // 2, 0))
ImageDraw.Draw(canvas).text(((W - im.width) // 2 + 16, H - 40), a.answer_note, font=f(20), fill=(200, 200, 200))
emit(canvas, 7)
# 3 左右に並べる（vlm_pick の最初の6秒の画面は除く）
skip = 6 * FPS
L = [x for x in io.get_reader(VID + "vlm_pick_t0.mp4")][skip:]; Rv = A[skip:]
n = max(len(L), len(Rv)); cw = 630; chh = int(cw * 640 / 1072)
for i in range(n):
    im = Image.new("RGB", (W, H), (0, 0, 0)); d = ImageDraw.Draw(im)
    d.text((60, 34), "同じ場面・同じ部品（緑：フランジ 28mm・本体 14mm、カメラには見えない）　実時間", font=f(22), fill=(170, 170, 170))
    for k, (V, lab, sub, col, res) in enumerate([(L, "SmolVLM で選んでそのまま掴む", "触らずに、カメラの推定だけで掴む点を選ぶ", (255, 120, 110), "掴み損ねた"),
                                                  (Rv, "SmolVLM ＋ 提案手法", "1回触って隠れたサイズを確かめてから掴む", (110, 170, 255), "掴めた")]):
        x = 6 + k * (cw + 8); fr = V[min(i, len(V) - 1)]
        im.paste(Image.fromarray(fr).resize((cw, chh)), (x, 170))
        d.text((x + 10, 84), lab, font=f(30, True), fill=col); d.text((x + 10, 128), sub, font=f(22, True), fill=col)
        if i > len(V) * 0.85:
            d.text((x + 10, 170 + chh + 20), "結果：" + res, font=f(32, True), fill=col)
    out.append_data(np.array(im))
emit(im, 2.5)
# 4 120回の結果
Rr = [json.loads(l) for fn in glob.glob("results/vlmcmp_*.jsonl") for l in open(fn)]
a0 = np.mean([r["notouch"]["success"] for r in Rr]) * 100; a1 = np.mean([r["touch"]["success"] for r in Rr]) * 100
by = lambda v, key: np.mean([r[key]["success"] for r in Rr if r["vis"] == v]) * 100
im, d = card(f"結果：{len({r['i'] for r in Rr})}配置 × 部品2つ＝{len(Rr)}回",
             [(f"SmolVLM で選んでそのまま掴む　{a0:.1f}%", RED), (f"SmolVLM ＋ 提案手法（1回触る）　{a1:.1f}%", ACC),
              "差 +19.2 ポイント（95% 区間 [+11.7, +26.7]、McNemar p < 0.001）",
              f"見えにくいほど差が大きい：見え方 10mm で {by(10, 'notouch'):.0f}% → {by(10, 'touch'):.0f}%",
              "",
              ("同じ観測から両方を評価（状態を複製して触らずに掴み、元の状態から触る方を続けた）", MUTED),
              ("SmolVLM が正しい部品を選べるかは別に評価（この比較は選択が正しい場合）", MUTED)],
             step="SmolVLM との組み合わせ")
emit(im, 12)
out.close()
print("docs/videos/vlm_compare.mp4")
