"""紹介動画（docs/videos/touch_selection.mp4、1分36秒）を作る。説明のカードと、配置64の2つの試行の動画を並べる。

使い方（リポジトリの直下で）:
  uv run python tools/make_video.py
材料：docs/figures/fig1_shapes.png・fig2_touch_rule.png、docs/videos/trial64_voi.mp4・trial64_unc.mp4
（試行の動画は view_trial.py 64 --rule voi --video ... で作り直せる）。フォントは Noto Sans CJK。
"""
import numpy as np, imageio.v2 as io
from PIL import Image, ImageDraw, ImageFont
FIG, VID = "docs/figures/", "docs/videos/"
W,H,FPS=1280,720,30
R="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; B="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
f=lambda s,b=False: ImageFont.truetype(B if b else R,s,index=0)
BG=(250,250,248); FG=(25,28,30); MUTED=(95,100,105); ACC=(30,95,170); RED=(175,45,45)
out=io.get_writer(VID+"touch_selection.mp4",fps=FPS,codec="libx264",quality=8,macro_block_size=16)
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
# 1 title
im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
d.text((80,210),"棚の奥で一部が隠れた部品を掴むための",font=f(48,True),fill=FG)
d.text((80,280),"触る点の選択",font=f(48,True),fill=FG)
d.text((80,370),"物理から学習した成功確率を、形の候補で平均する",font=f(30),fill=MUTED)
d.text((80,450),"MuJoCo ／ Unitree G1 ＋ Dex3",font=f(26),fill=MUTED)
d.text((80,600),"Physical AI 講座 最終課題　Nishina",font=f(24),fill=MUTED)
emit(im,5)
# 2 problem
emit(card("問題：奥が見えないと、どこを挟めばよいか分からない",
  ["天板に隠れて、カメラにはフランジの手前しか見えない。下の3つは見た目が同じ。"],
  img=FIG+"fig1_shapes.png",img_box=(60,200,1220,670),step="1 / 5　問題",
  foot="手前に低いフランジ（4mm）、奥に掴める本体。フランジと本体の長さは未知。フランジだけ挟んでも持ち上がらない。"),12)
# 3 method
emit(card("方法：触った後に「掴める確率」が最も上がる所を触る",
 ["① カメラ（領域分割＋深度）で、形の候補を200個つくる",
  "② 候補ごとの把持成功確率を小さなニューラルネットで出し、候補で平均",
  "　　（摩擦・質量・指のゲインを振った物理シミュレーション 8,000回で学習）",
  "③ 触る点：触った後に得られる最良の成功確率の期待値が最大の点",
  ("　　比較：形の不確かさが最大の点（先行研究の基準）、ランダム",MUTED),
  "④ 中指で上から軽く触る（約10g）→ 天板の下のまま、そのまま掴む",
  "",
  ("u* = argmax_u  Σ_o P(o | u, b) · max_g P(成功 | g, b_{u,o})",ACC)],step="2 / 5　方法"),14)
# 4 comparison intro
emit(card("同じ配置で、触る点の選び方だけを変える",
 ["配置64：見えている長さ 10mm、本体は手前から 15〜31mm（カメラからは見えない）",
  "",("左：提案手法 …… 本体の奥より先（43mm）を触って「外れ」",ACC),
  ("　　→ 本体はそこまで奥にないと分かり、手前の縁のばらつき 5.9 → 2.9mm",ACC),
  ("右：不確かさ最大 …… 本体の真ん中（23mm）を触って「当たり」",RED),
  ("　　→ 本体はあると分かるが、手前の縁のばらつきは 4.3mm 残る",RED),
  "","赤：触る点　青：掴む点　黄：形の候補ごとの本体の手前の縁（200個）",
  "右上の小窓：ロボットの頭部カメラの映像"],step="3 / 5　比較"),9)
# 5 side-by-side video
A=[x for x in io.get_reader(VID+"trial64_voi.mp4")]; Bv=[x for x in io.get_reader(VID+"trial64_unc.mp4")]
n=max(len(A),len(Bv)); cw=630; chh=int(cw*544/960)
for i in range(n):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    d.text((60,34),"3 / 5　比較（実時間）",font=f(22),fill=MUTED)
    for k,(V,lab,col,res) in enumerate([(A,"提案手法",ACC,"掴めた"),(Bv,"不確かさ最大",RED,"掴み損ねた（部品が落ちる）")]):
        x=10+k*(cw+0); fr=V[min(i,len(V)-1)]
        im.paste(Image.fromarray(fr).resize((cw,chh)),(x,170))
        d.text((x+10,110),lab,font=f(34,True),fill=col)
        if i>len(V)*0.80: d.text((x+10,170+chh+20),"結果："+res,font=f(32,True),fill=col)
    out.append_data(np.array(im))
emit(im,2.5)
emit(card("提案手法はどこを触っているのか（240配置の1回目）",
 [("提案手法：本体の外（手前の縁の手前側、または奥の縁の先）59%、本体の中ほど 5%",ACC),
  ("　　→ 当たり方（指の腹／フランジ／外れ）で「縁がどちら側にあるか」が分かる",ACC),
  ("不確かさ最大：本体の外 18%、指先の中央が本体に当たる 73%",RED),
  ("　　→ 本体があることは分かるが、掴む位置を決める手前の縁は絞れない",RED),
  "",
  "配置64は「奥の縁の先」を触った例。多くの配置では手前の縁の手前側を触る。",
  "どちらも「触った結果で、掴む点の選び方が最も変わる所」を選んだ結果。"],step="3 / 5　比較"),10)
# 6 results
emit(card("結果：240配置で、1回触った時点の把持成功率",
  [],img=FIG+"fig2_touch_rule.png",img_box=(60,130,1220,560),step="4 / 5　結果",
  foot=None),1/FPS)
im=card("結果：240配置で、1回触った時点の把持成功率",[],img=FIG+"fig2_touch_rule.png",img_box=(60,120,1220,545),step="4 / 5　結果")
d=ImageDraw.Draw(im)
d.text((70,565),"提案手法 95.4%　不確かさ最大 88.8%　ランダム 88.8%",font=f(30,True),fill=FG)
d.text((70,612),"同じ配置どうしの差 +6.7ポイント（95%区間 [+2.1, +11.3]、McNemar p = 0.005）。見えにくいほど差が大きい（10mmで +15.0）。",font=f(22),fill=MUTED)
d.text((70,648),"触らずに掴むと 72.5%。失敗した把持の9割で部品が10mm以上動く（掴み直しが難しい）。",font=f(22),fill=MUTED)
emit(im,14)
# 7 other findings + conclusion
emit(card("わかったこと",
 ["・触覚で形を補うときは、形を正確に復元する所ではなく、",
  "　掴むことに効く所を触るべき（1回で成功率はほぼ上限の約97%）",
  "・成功確率が閾値を超えたら触るのをやめる規則は、毎回1回触るのと差がなかった",
  "・触った後に腕を天板の外へ戻さず、そのまま掴むと、",
  "　掴み終わるまでの時間が約14%（1回）／27%（2回）短くなった",
  ("・限界：シミュレーションのみ（実機は未検証）、形の型は既知で長さ2つが未知",MUTED),
  "","コード・学習済みモデル・全結果：",
  ("github.com/Nishina-N/Physics-Informed-Touch-Selection",ACC)],step="5 / 5　まとめ"),14)
out.close()
