"""English figure for the SmolVLM combination (paper/figures/vlm_compare.png), from the last frames of
docs/videos/vlm_pick_t0.mp4 (no touch) and docs/videos/vlm_pick.mp4 (one touch). Both are the same scene and the same
SmolVLM answer ("2", the green part). The Japanese labels of the side panels are cropped away and redrawn in English.

Usage (from the repository root): python paper/make_vlm_figure.py
"""
import os
import imageio.v2 as io
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(__file__), "..")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
f = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s, index=0)
INK, MUTED, BLUE, RED, BG = (25, 28, 30), (95, 100, 105), (30, 95, 170), (175, 45, 45), (255, 255, 255)
MAIN, HEAD, CLOSE = (16, 72, 736, 612), (752, 96, 1056, 324), (752, 364, 1056, 592)   # regions in the 1072x640 frame


def panel(video, title, sub, result, col):
    fr = Image.fromarray([x for x in io.get_reader(os.path.join(ROOT, "docs/videos", video))][-1])
    W, H = 1060, 700
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.text((10, 6), title, font=f(34, True), fill=col)
    d.text((10, 52), sub, font=f(22), fill=col)
    im.paste(fr.crop(MAIN), (10, 100))
    for k, (box, lab) in enumerate([(HEAD, "Head camera (board hides most of the part)"), (CLOSE, "Close-up (board drawn translucent)")]):
        y = 100 + k * 275
        d.text((746, y), lab, font=f(15), fill=MUTED)
        im.paste(fr.crop(box), (746, y + 24))
    d.text((10, 648), result, font=f(30, True), fill=col)
    return im


def main():
    L = panel("vlm_pick_t0.mp4", "SmolVLM pick, grasp without touching", "Grasp point from the camera estimate (predicted success 0.57)",
              "Result: failure", RED)
    R = panel("vlm_pick.mp4", "SmolVLM pick + ours (one touch)", "Touch hits the flange; the body is further back (predicted 0.92)",
              "Result: success (lifted 5.9 cm)", BLUE)
    out = Image.new("RGB", (L.width * 2 + 20, L.height), BG)
    out.paste(L, (0, 0)); out.paste(R, (L.width + 20, 0))
    path = os.path.join(ROOT, "paper/figures/vlm_compare.png")
    out.save(path); print(path)


if __name__ == "__main__":
    main()
