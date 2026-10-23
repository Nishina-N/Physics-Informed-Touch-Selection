"""紹介動画（docs/videos/touch_selection.mp4、3分以内）を作る。説明のカードと、配置64の2つの試行の動画を並べる。

使い方（リポジトリの直下で）:
  uv run python tools/make_video.py [--op 講座のオープニング.mp4] [--ed 講座のエンディング.mp4]
--op・--ed を渡すと、講座から配られたオープニングとエンディング（各約3秒）を最初と最後につなぐ（どちらもリポジトリには入れていない）。
材料：docs/figures/fig0_pipeline.png・fig1_shapes.png・fig2_touch_rule.png、docs/videos/trial64_voi.mp4・trial64_unc.mp4（全体）・trial64_voi_close.mp4・trial64_unc_close.mp4（手元）・trial64_*_macro.mp4（指先）
（試行の動画は view_trial.py 64 --rule voi --video ...、手元は --close、指先は --macro を足して作り直せる）。フォントは Noto Sans CJK。
"""
import argparse
import numpy as np, imageio.v2 as io
from PIL import Image, ImageDraw, ImageFont
FIG, VID = "docs/figures/", "docs/videos/"
U = "形の不確かさ最大の場所を触って精度向上"     # 比較の規則（unc）の呼び名
W,H,FPS=1280,720,30
R="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; B="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
f=lambda s,b=False: ImageFont.truetype(B if b else R,s,index=0)
BG=(250,250,248); FG=(25,28,30); MUTED=(95,100,105); ACC=(30,95,170); RED=(175,45,45)
out=io.get_writer(VID+"touch_selection.mp4",fps=FPS,codec="libx264",quality=8,macro_block_size=16)
ap=argparse.ArgumentParser(); ap.add_argument("--op"); ap.add_argument("--ed"); ARGS=ap.parse_args()
def clip(fn):
    """講座のオープニング・エンディングを、この動画の大きさに合わせてそのままつなぐ。"""
    for fr in io.get_reader(fn):
        out.append_data(np.array(Image.fromarray(fr).resize((W,H),Image.LANCZOS)))
def emit(img,sec):
    a=np.array(img.convert("RGB"))
    for _ in range(int(sec*FPS)): out.append_data(a)
def card(title,lines=(),img=None,img_box=None,foot=None,step=None):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    if step: d.text((60,34),step,font=f(22),fill=MUTED)
    d.text((60,64),title,font=f(40,True),fill=FG)
    y=140
    for ln in lines:
        col=FG; s=ln
        if isinstance(ln,tuple): s,col=ln
        d.text((70,y),s,font=f(28),fill=col); y+=48
    if img is not None:
        x0,y0,x1,y1=img_box; p=Image.open(img).convert("RGB"); p.thumbnail((x1-x0,y1-y0))
        im.paste(p,(x0+(x1-x0-p.width)//2,y0+(y1-y0-p.height)//2))
    if foot: d.text((60,H-50),foot,font=f(20),fill=MUTED)
    return im
if ARGS.op: clip(ARGS.op)
# 1 title
im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
d.text((80,210),"棚の奥で一部が隠れた部品を掴むための",font=f(48,True),fill=FG)
d.text((80,280),"触る点の選択",font=f(48,True),fill=FG)
d.text((80,370),"物理から学習した成功確率を、形の候補で平均する",font=f(30),fill=MUTED)
d.text((80,450),"MuJoCo ／ Unitree G1 ＋ Dex3",font=f(26),fill=MUTED)
d.text((80,600),"Physical AI 応用1講座 最終課題",font=f(24),fill=MUTED)
emit(im,4)
# 2 problem
emit(card("問題：奥が見えないと、どこを挟めばよいか分からない",
  ["天板に隠れて、カメラにはフランジの手前しか見えない。下の3つは見た目が同じ。"],
  img=FIG+"fig1_shapes.png",img_box=(60,200,1220,670),step="1 / 6　問題",
  foot="手前に低いフランジ（4mm）、奥に掴める本体。フランジと本体の長さは未知。フランジだけ挟んでも持ち上がらない。"),12)
# 3a 学習と推論の流れ（ブロック図、tools/fig_pipeline.py）
emit(Image.open(FIG+"fig0_pipeline.png").convert("RGB").resize((W,H),Image.LANCZOS),13)
# 3 method
emit(card("方法：触った後に「掴める確率」が最も上がる所を触る",
 ["① カメラ（領域分割＋深度）で、形の候補を200個つくる",
  "② 候補ごとの把持成功確率を小さなニューラルネットで出し、候補で平均",
  "　　（摩擦・質量・指のゲインを振った物理シミュレーション 8,000回で学習）",
  "③ 触る点：触った後に得られる最良の成功確率の期待値が最大の点",
  (f"　　比較：{U}（先行研究の基準）、ランダム",MUTED),
  "④ 中指で上から軽く触る（約10g）→ 天板の下のまま、そのまま掴む",
  "",
  ("u* = argmax_u  Σ_o P(o | u, b) · max_g P(成功 | g, b_{u,o})",ACC)],step="2 / 6　方法"),9)
# 4 comparison intro
emit(card("同じ配置で、触る点の選び方だけを変える",
 ["配置64：見えている長さ 10mm、本体は手前から 15〜31mm（カメラからは見えない）",
  ("左：提案手法（掴める確率が最も上がる場所を触る）",ACC),
  ("　　…… 本体の奥より先（43mm）を触って「外れ」",ACC),
  ("　　→ 本体はそこまで奥にないと分かり、手前の縁のばらつき 5.9 → 2.9mm",ACC),
  (f"右：{U}",RED),
  ("　　…… 本体の真ん中（23mm）を触って「当たり」",RED),
  ("　　→ 本体はあると分かるが、手前の縁のばらつきは 4.3mm 残る",RED),
  "","赤：触る点　青：掴む点　黄：形の候補ごとの本体の手前の縁（200個）",
  "左：ロボット全体を横から　右の小窓：上が頭部カメラ、下が部品のまわりの拡大"],step="3 / 6　比較"),9)
# 5 side-by-side video
A=[x for x in io.get_reader(VID+"trial64_voi.mp4")]; Bv=[x for x in io.get_reader(VID+"trial64_unc.mp4")]
n=max(len(A),len(Bv)); cw=630; chh=int(cw*640/1072)            # 試行の動画（1072×640、黒背景）を縮めて左右に並べる
for i in range(n):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.text((60,34),"3 / 6　比較（実時間）",font=f(22),fill=MUTED)
    for k,(V,lab,sub,col,res) in enumerate([(A,"提案手法","掴める確率が最も上がる場所を触る",ACC,"掴めた"),
                                             (Bv,"比較",U,RED,"掴み損ねた（部品が落ちる）")]):
        x=6+k*(cw+8); fr=V[min(i,len(V)-1)]
        im.paste(Image.fromarray(fr).resize((cw,chh)),(x,170))
        d.text((x+10,84),lab,font=f(32,True),fill=col); d.text((x+10,128),sub,font=f(24,True),fill=col)
        if i>len(V)*0.80: d.text((x+10,170+chh+20),"結果："+res,font=f(32,True),fill=col)
    out.append_data(np.array(im))
emit(im,2.5)
# 5b 手元のアップ（横から、部品が机の手前に落ちるところまで入る）、5c 指先のアップ（どこを触ったか）
for tag,head in (("close","3 / 6　比較：手元のアップ（横から、実時間）"),
                 ("macro","3 / 6　比較：指先のアップ（実時間）　赤：触る点　青：掴む点　黄：形の候補ごとの本体の手前の縁")):
  A=[x for x in io.get_reader(VID+f"trial64_voi_{tag}.mp4")]; Bv=[x for x in io.get_reader(VID+f"trial64_unc_{tag}.mp4")]
  n=max(len(A),len(Bv)); cw=560; chh=int(cw*560/640)
  for i in range(n):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.text((60,34),head,font=f(22),fill=MUTED)
    for k,(V,lab,sub,col,res) in enumerate([(A,"提案手法","掴める確率が最も上がる場所を触る",ACC,"掴めた"),
                                             (Bv,"比較",U,RED,"掴み損ねた（部品が落ちる）")]):
        x=40+k*(cw+80); fr=V[min(i,len(V)-1)]
        im.paste(Image.fromarray(fr).resize((cw,chh)),(x,165))
        d.text((x,80),lab,font=f(30,True),fill=col); d.text((x,122),sub,font=f(22,True),fill=col)
        if i>len(V)*0.80: d.text((x,165+chh+8),"結果："+res,font=f(28,True),fill=col)
    out.append_data(np.array(im))
  emit(im,2.5)
emit(card("提案手法はどこを触っているのか",
 [("提案手法：本体の外（手前の縁の手前側、または奥の縁の先）59%、本体の中ほど 5%",ACC),
  ("　　→ 当たり方（指の腹／フランジ／外れ）で「縁がどちら側にあるか」が分かる",ACC),
  (f"{U}：本体の外 18%、指先の中央が本体に当たる 73%",RED),
  ("　　→ 本体があることは分かるが、掴む位置を決める手前の縁は絞れない",RED),
  "",
  "今回動画で示した配置64は「奥の縁の先」を触った例。",
  "多くの配置では手前の縁の手前側を触る。",
  "どちらも「触った結果で、掴む点の選び方が最も変わる所」を選んだ結果。"],step="3 / 6　比較"),10)
# 6 results
emit(card("結果：240配置で、1回触った時点の把持成功率",
  [],img=FIG+"fig2_touch_rule.png",img_box=(60,130,1220,560),step="4 / 6　結果",
  foot=None),1/FPS)
im=card("結果：240配置で、1回触った時点の把持成功率",[],img=FIG+"fig2_touch_rule.png",img_box=(60,120,1220,545),step="4 / 6　結果")
d=ImageDraw.Draw(im)
d.text((70,565),f"提案手法 95.4%　{U} 88.8%　ランダム 88.8%",font=f(28,True),fill=FG)
d.text((70,608),"同じ配置どうしの差 +6.7ポイント（95%区間 [+2.1, +11.3]、McNemar p = 0.005）。",font=f(21),fill=MUTED)
d.text((70,638),"見えにくいほど差が大きい（10mmで +15.0）。触らずに掴むと 72.5%。",font=f(21),fill=MUTED)
d.text((70,668),"失敗した把持の9割で部品が10mm以上動く（掴み直しが難しい）。",font=f(21),fill=MUTED)
emit(im,12)
# 6b SmolVLM との組み合わせ（材料は vlm_pick.py の2本と results/vlmcmp_*・vlmpoint_*、tools/make_vlm_video.py と同じ）
emit(card("SmolVLM で部品を選び、触って確かめてから掴む",
 ["棚の下に色とサイズの違う部品が2つ（どちらも手前しか見えない）",
  "① SmolVLM-500M：指示文と画像から、取る部品を番号で選ぶ",
  "② 提案手法：1回触ってサイズを確かめ、掴む点を選ぶ",
  ("比較：② を省き、カメラの推定だけで掴む",RED),
  ("手の経路はどちらも同じ逆運動学。違うのは触るかどうかだけ",MUTED)],step="5 / 6　SmolVLM との組み合わせ"),7)
VA=[x for x in io.get_reader(VID+"vlm_pick.mp4")]; VB=[x for x in io.get_reader(VID+"vlm_pick_t0.mp4")]
hh=H-50; p=Image.fromarray(VA[0]).resize((int(hh*VA[0].shape[1]/VA[0].shape[0]),hh))
im=Image.new("RGB",(W,H),BG); im.paste(p,((W-p.width)//2,0))
ImageDraw.Draw(im).text(((W-p.width)//2+16,H-40),"（SmolVLM-500M が GPU で答えた「2」。左の触らない側も同じ場面・同じ答えで動かした）",font=f(20),fill=MUTED)
emit(im,5)
skip=6*FPS; L=VB[skip:]; Rv=VA[skip:]; n=max(len(L),len(Rv)); cw=630; chh=int(cw*640/1072)
for i in range(n):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.text((60,34),"5 / 6　同じ場面・同じ部品（緑：フランジ 28mm・本体 13mm、カメラには見えない）　実時間",font=f(22),fill=MUTED)
    for k,(V,lab,sub,col,res) in enumerate([(L,"SmolVLM で選んでそのまま掴む","触らずに、カメラの推定だけで掴む点を選ぶ",RED,"掴み損ねた"),
                                             (Rv,"SmolVLM ＋ 提案手法","1回触って隠れたサイズを確かめてから掴む",ACC,"掴めた")]):
        x=6+k*(cw+8); fr=V[min(i,len(V)-1)]
        im.paste(Image.fromarray(fr).resize((cw,chh)),(x,170))
        d.text((x+10,84),lab,font=f(30,True),fill=col); d.text((x+10,128),sub,font=f(22,True),fill=col)
        if i>len(V)*0.85: d.text((x+10,170+chh+20),"結果："+res,font=f(32,True),fill=col)
    out.append_data(np.array(im))
emit(im,1.5)
emit(card("結果：部品2つの場面で120回（60配置 × 2）",
 [("SmolVLM-2.2B が掴む位置も選ぶ（長さの範囲を文で与える）　61.7%",RED),
  ("SmolVLM で選び、触らずに予測器で掴む　75.8%",FG),
  ("SmolVLM で選び、1回触ってから掴む（提案手法）　95.0%",ACC),
  "触ってから掴むと +19.2 ポイント（McNemar p < 0.001）",
  ("部品選びは、指示文と色の並びを変えた3回とも正しかった",MUTED),
  ("掴む位置を VLM に任せ、長さの知識も与えないと 0.8%",MUTED)],step="5 / 6　SmolVLM との組み合わせ"),9)
# 7 other findings + conclusion
emit(card("わかったこと",
 ["・触覚で形を補うときは、形を正確に復元する所ではなく、",
  "　掴む点の選び方が最も変わる所を触る（1回で成功率はほぼ上限の約97%）",
  "・SmolVLM で部品を選ぶ場合も、掴む前に触ると 75.8% → 95.0%",
  "・成功確率が閾値を超えたら触るのをやめる規則は、毎回1回触るのと差がなかった",
  "・触った後に腕を天板の外へ戻さず、そのまま掴むと、",
  "　掴み終わるまでの時間が約14%（1回）／27%（2回）短くなった",
  ("・限界：シミュレーションのみ（実機は未検証）、形の型は既知で長さ2つが未知",MUTED),
  "","コード・学習済みモデル・全結果：",
  ("github.com/Nishina-N/Physics-Informed-Touch-Selection",ACC)],step="6 / 6　まとめ"),12)
if ARGS.ed: clip(ARGS.ed)
out.close()
