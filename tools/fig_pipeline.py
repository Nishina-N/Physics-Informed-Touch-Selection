"""学習と推論の流れのブロック図（docs/figures/fig0_pipeline.png、1920×1080）を作る。レポートと動画で使う。

使い方（リポジトリの直下で）:
  uv run python tools/fig_pipeline.py
  FIG_LANG=en uv run python tools/fig_pipeline.py     # 論文用（英語）→ paper/figures/pipeline.png
SVG を書いて Chromium（playwright）で PNG にする。フォントは Noto Sans CJK JP。
"""
import os
from playwright.sync_api import sync_playwright
EN = os.environ.get("FIG_LANG") == "en"
T = lambda ja, en: en if EN else ja

ACC, FG, MUTED, LINE = "#1e5faa", "#191c1e", "#5f6469", "#9aa0a6"
LEARN_BG, HI_BG = "#eef3f9", "#fff6e0"
el = []                                          # SVG の要素


def box(x, y, w, h, title, lines, src=None, bg="#ffffff", edge=LINE, ew=1.5):
    el.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{bg}' stroke='{edge}' stroke-width='{ew}'/>")
    el.append(f"<text x='{x+14}' y='{y+30}' font-size='19' font-weight='700' fill='{FG}'>{title}</text>")
    for i, s in enumerate(lines):
        el.append(f"<text x='{x+14}' y='{y+58+i*22}' font-size='15' fill='{FG}'>{s}</text>")
    if src:
        el.append(f"<text x='{x+w-12}' y='{y+h-10}' font-size='12' fill='{MUTED}' text-anchor='end'>{src}</text>")


def arrow(d, label=None, lx=0, ly=0, color=FG, dash=False):
    el.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.2' marker-end='url(#a{'b' if color == ACC else ''})'"
              + (" stroke-dasharray='6 4'" if dash else "") + "/>")
    if label:
        el.append(f"<text x='{lx}' y='{ly}' font-size='15' fill='{color}'>{label}</text>")


# 見出し
el.append(f"<text x='40' y='44' font-size='28' font-weight='700' fill='{FG}'>{T('学習と推論の流れ', 'Learning and inference')}</text>")
el.append(f"<text x='{T(290, 370)}' y='44' font-size='17' fill='{MUTED}'>{T('学習は事前に1回。推論は配置ごとに ①〜⑦ を行う（触る回数ぶん ③〜⑤ を繰り返す）', 'Learning runs once offline. Inference runs (1)–(7) per layout; (3)–(5) repeat for each touch.')}</text>")

# 学習（上段）
el.append(f"<text x='40' y='88' font-size='18' font-weight='700' fill='{ACC}'>{T('学習（事前に1回、物理シミュレーションの試行から）', 'Learning (once, from physics-simulation trials)')}</text>")
LX, LW, LY, LH = [40, 345, 650, 955], 270, 100, 150
box(LX[0], LY, LW, LH, T("データを集める", "Collect grasps"), [T("配置と物理の値を乱数で引く", "Sample layout and physics"), T("（摩擦・質量・指のゲイン）", "(friction, mass, finger gain)"), T("挟む点を振って真の形で掴む", "Grasp the true shape at"), T("MuJoCo で 8,000 回", "random points: 8,000 trials")], "gen_grasp_data.py", LEARN_BG, ACC)
box(LX[1], LY, LW, LH, T("特徴と成否を記録", "Record features"), [T("入力 z：本体の縁から挟む点", "Input z: 6 features, e.g."), T("　までの距離など 6 つ", "  distances to body edges"), T("出力 S：4cm 持ち上げて", "Output S: lifted 4 cm and"), T("　2 秒保持できたか", "  held for 2 s")], None, LEARN_BG, ACC)
box(LX[2], LY, LW, LH, T("成功確率の予測器を学習", "Train success predictor"), [T("小さなニューラルネット", "Small neural network"), T("（MLP 64×64）", "(MLP 64×64)"), T("f_φ(z) → 成功確率", "f_φ(z) → success prob."), T("摩擦などはデータの中で平均", "averaged over physics")], "train_predictor.py", LEARN_BG, ACC)
box(LX[3], LY, LW, LH, T("学習済みの予測器 f_φ", "Trained predictor f_φ"), [T("確認用データで", "On held-out data:"), T("　正解率 94%", "  accuracy 94%"), T("　Brier スコア 0.044", "  Brier score 0.044")], "predictor.pkl", LEARN_BG, ACC, 2.5)
for i in range(3):
    arrow(f"M{LX[i]+LW},{LY+LH/2} H{LX[i+1]-4}")

# 推論（下段）
IX, IW, R1, R2, IH = [40, 285, 530, 775, 1020], 210, 330, 545, 140
el.append(f"<text x='40' y='312' font-size='18' font-weight='700' fill='{FG}'>{T('推論（配置ごと、実行時）', 'Inference (per layout, at run time)')}</text>")
box(IX[0], R1, IW, IH, T("① 頭部カメラで見る", "(1) Look"), [T("領域分割と深度画像", "Head camera: mask"), T("奥は天板で見えない", "and depth; board")] + ([] if not EN else ["hides the back"]), "perception.py")
box(IX[1], R1, IW, IH, T("② 形の候補をつくる", "(2) Candidates"), [T("候補（粒子）を 200 個", "200 particles:"), T("位置・向き・", "pose, flange length,"), T("フランジ長・本体長", "body length")], "perception.py")
box(IX[2], R1, IW, IH, T("③ 触る点を選ぶ", "(3) Choose touch"), [T("触った後に得られる", "Max. expected best"), T("最良の成功確率の", "grasp success"), T("期待値が最大の点", "after the touch")], "partfilter.py", HI_BG, ACC, 2.5)
box(IX[3], R1, IW, IH, T("④ 中指で軽く触る", "(4) Touch lightly"), [T("上から約 10g で触れる", "Middle finger, ~10 g"), T("観測：外れ／本体／", "Outcome: miss / body /"), T("フランジ（4 通り）", "flange (4 classes)")], "shelf.py")
box(IX[4], R1, IW, IH, T("⑤ 候補を更新", "(5) Update"), [T("観測の起こりやすさで", "Reweight by outcome"), T("重みを付けて引き直す", "likelihood, resample")], "partfilter.py")
box(IX[4], R2, IW, IH, T("⑥ 掴む点を選ぶ", "(6) Choose grasp"), [T("候補で平均した", "Max. success averaged"), T("成功確率が最大の点", "over candidates")], "partfilter.py", HI_BG, ACC, 2.5)
box(IX[3], R2, IW, IH, T("⑦ 掴んで持ち上げる", "(7) Grasp and lift"), [T("天板の下のまま移り、", "Move under the board,"), T("中指と親指で挟む", "pinch with 2 fingers")], "shelf.py")
for i in range(4):
    arrow(f"M{IX[i]+IW},{R1+IH/2} H{IX[i+1]-4}")
arrow(f"M{IX[4]+160},{R1+IH} V{R2-4}")                                     # ⑤ → ⑥
arrow(f"M{IX[4]},{R2+IH/2} H{IX[3]+IW+4}")                                  # ⑥ → ⑦
arrow(f"M{IX[4]+50},{R1+IH} V{R1+IH+40} H{IX[2]+IW/2} V{R1+IH+4}",          # ⑤ → ③（繰り返し）
      T("もう一度触るなら ③ へ（主な評価は 1 回）", "To touch again, back to (3) (main evaluation: 1 touch)"), IX[2] + IW / 2 + 14, R1 + IH + 32)
# 予測器 → ③・⑥
arrow(f"M{LX[3]+60},{LY+LH} V{284} H{IX[2]+IW/2} V{R1-4}", T("f_φ を使う", "uses f_φ"), IX[2] + IW / 2 + 10, 300, ACC)
arrow(f"M{LX[3]+LW},{LY+LH/2} H{IX[4]+IW+20} V{R2+IH/2} H{IX[4]+IW+4}", None, 0, 0, ACC, True)   # 予測器の右の辺から出て、右端を下り ⑥ の右の辺へ
# 凡例
el.append(f"<rect x='40' y='{R2+8}' width='22' height='16' rx='3' fill='{HI_BG}' stroke='{ACC}' stroke-width='2'/>")
el.append(f"<text x='72' y='{R2+22}' font-size='15' fill='{FG}'>{T('学習した予測器 f_φ を使う所', 'Steps that use the learned predictor f_φ')}</text>")
el.append(f"<text x='40' y='{R2+52}' font-size='15' fill='{FG}'>{T('候補ごとに f_φ で成功確率を出し、', 'f_φ gives success per candidate,')}</text>")
el.append(f"<text x='40' y='{R2+74}' font-size='15' fill='{FG}'>{T('200 個の候補で平均する', 'averaged over 200 candidates')}</text>")
el.append(f"<text x='40' y='{R2+106}' font-size='15' fill='{MUTED}'>{T('比較：③ で「形の不確かさ最大の場所」', 'Baselines change only (3): touch')}</text>")
el.append(f"<text x='40' y='{R2+128}' font-size='15' fill='{MUTED}'>{T('　またはランダムに触る（他は同じ）', 'max. shape uncertainty, or random')}</text>")
el.append(f"<text x='40' y='{R2+152}' font-size='15' fill='{MUTED}'>{T('成功＝4cm 以上持ち上がり、2 秒保持できた', 'Success: lifted ≥ 4 cm, held 2 s')}</text>")

svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1280 720' width='1280' height='720' "
       f"font-family='Noto Sans CJK JP'><defs>"
       f"<marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='{FG}'/></marker>"
       f"<marker id='ab' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='{ACC}'/></marker>"
       f"</defs><rect width='1280' height='720' fill='#fafaf8'/>" + "".join(el) + "</svg>")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
    pg.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
    pg.screenshot(path=T("docs/figures/fig0_pipeline.png", "paper/figures/pipeline.png"))
    b.close()
print(T("docs/figures/fig0_pipeline.png", "paper/figures/pipeline.png"))
