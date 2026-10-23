"""コードの読み方（docs/code_guide.html）を作る。左に今のコードの抜粋、右に解説を、学習と推論で実行する順に並べる。

使い方（リポジトリの直下で）:
  uv run python tools/make_walkthrough.py
コードは実行時のファイルから関数の名前で切り出す（ast）ので、コードを直したら作り直せば解説の左側も追従する。
解説（右側）はこのファイルに書いてある。関数の中身が変わったら、解説も合わせて直す。
GitHub へのリンクは、作った時点のコミットの行番号を指す。
"""
import ast, html, os, subprocess, sys
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPO = "https://github.com/Nishina-N/Physics-Informed-Touch-Selection"
OUT = os.path.join(ROOT, "docs", "code_guide.html")


def src_lines(file):
    return open(os.path.join(ROOT, file), encoding="utf-8").read().split("\n")


def find(file, qualname=None, start=None, end=None):
    """切り出す行の範囲（1始まり、両端を含む）。qualname は関数・クラスの名前（ShelfSim.touch など）。
    start・end は、その範囲の中で最初に現れる文字列（その行から／その行まで）。"""
    lines = src_lines(file)
    a, b = 1, len(lines)
    if qualname:
        node = ast.parse("\n".join(lines))
        for part in qualname.split("."):
            node = next(n for n in ast.iter_child_nodes(node)
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
        a, b = node.lineno, node.end_lineno
    if start:
        a = next(i for i in range(a, b + 1) if start in lines[i - 1])
    if end:
        b = next(i for i in range(a, b + 1) if end in lines[i - 1])
    return a, b


def code_html(file, a, b):
    text = "\n".join(src_lines(file)[a - 1:b])
    fmt = HtmlFormatter(linenos="table", linenostart=a, cssclass="hl", wrapcode=True)
    return highlight(text, PythonLexer(), fmt)


# ---------------------------------------------------------------------------------------------
# 解説。part：どの段か、no：図の番号、title：見出し、src：(ファイル, 関数, 開始の文字列, 終了の文字列)
# role：何をするか（1〜2文）、io：入力と出力、points：読みどころ、note：補足（灰色）
# ---------------------------------------------------------------------------------------------
CH = []


def ch(part, no, title, src, role, io=(), points=(), note=None):
    CH.append(dict(part=part, no=no, title=title, src=src, role=role, io=io, points=points, note=note))


# ===== 学習 ==================================================================================
ch("learn", "学習 0", "場面をつくる：部品はフランジと本体をつないだ1つの剛体",
   ("build_scene.py", "build", "# 立方体（自由に動く）", "return spec"),
   "MuJoCo Menagerie の G1（Dex3 ハンド付き）に、机・頭部カメラ・部品を足してシーンを組み立てる関数のうち、部品を作る部分です。"
   "<code>part=(fl, L)</code> を渡すと、手前に高さ 4mm・長さ fl のフランジ、奥に幅と高さ 20mm・長さ L の本体をつないだ部品になります。",
   io=["入力：部品の幅 <code>s</code>（20mm）、<code>part=(フランジの長さ, 本体の長さ)</code>",
       "出力：<code>mujoco.MjSpec</code>（まだコンパイル前のモデル）。部品の名前は <code>cube_20mm</code>、フランジは <code>cube_20mm_flange</code>"],
   points=["フランジと本体は同じ <code>body</code> に付けた2つの箱です。1つの剛体なので、フランジだけ挟んでも本体ごと動きます。",
           "フランジの箱は <code>part[0] + 0.001</code> と 1mm 長くして本体に重ね、隙間ができないようにしています。",
           "<code>condim=4</code> はねじり摩擦を入れる設定です。既定の 3 だと、挟んだ軸のまわりに部品が回ってしまいます。",
           "摩擦 0.5 はここでの初期値です。学習データと実験では、試行ごとに 0.3〜1.0 の値に置き換えます（次の節）。"],
   note="関数の前半（この抜粋より上）では、骨盤を世界に固定し、頭部カメラ D435i を胴体に付け、楕円の摩擦錐と impratio=10 で把持の滑りを抑える設定をしています。")

ch("learn", "学習 0", "見え方から天板の前端の位置を決める",
   ("shelf.py", "board_front_for_visible"),
   "「フランジの上面が手前から何 mm 見えるか（見え方）」を指定すると、それに合う天板の前端の x を返します。実験では見え方を 10・15・20・25mm の4段階に振ります。",
   io=["入力：見え方 <code>visible</code>[m]、見える面の高さ <code>width</code>（フランジなら 4mm）",
       "出力：天板の前端（ロボット側の縁）の x[m]"],
   points=["頭部カメラ <code>CAM_P</code> から、天板の前端の下の角を通る光線を引きます。その光線がフランジの上面の高さ <code>zt</code> に届く点が、見える範囲の奥の端です。",
           "奥の端を <code>X_FRONT + visible</code> に置きたいので、相似な三角形の比から前端の x を逆算しています。",
           "<code>CAM_P</code> は開始姿勢のシミュレーションで測ったカメラの位置です。"])

ch("learn", "学習 ①", "物理の値を乱数で引く",
   ("gen_grasp_data.py", None, "W = 0.020", "return dict(mu=mu"),
   "1回の把持ごとに、摩擦・部品の質量・指の制御ゲインを乱数で変えます。これらは特徴量には入れないので、予測器は「これらがばらついたときの平均的な成功確率」を学びます。",
   io=["入力：シミュレーション <code>s</code>、乱数 <code>rng</code>",
       "出力：引いた値 <code>dict(mu, mass_scale, gain_scale)</code>（記録用）"],
   points=["摩擦 μ は 0.3〜1.0。MuJoCo は接触する2つの形状の摩擦の大きい方を使うので、指と部品の両方に同じ値を入れます。",
           "質量は 0.7〜1.3 倍。慣性も同じ倍率で変えます。",
           "指のゲインは 0.8〜1.2 倍。位置制御なので <code>gainprm</code> と <code>biasprm</code> を同じ倍率で変え、目標角の位置は変えずに硬さだけを変えます。",
           "<code>mj_setConst</code> で定数を計算し直します。これを忘れると動きが不安定になり、成功率が見かけ上下がりました（コメントに残してあります）。"],
   note="上の定数は、形の範囲（フランジ 10〜30mm、本体 12〜20mm）、見え方の4段階、物理の値の範囲です。推論側の <code>run_branch.py</code> も同じ範囲を使います。")

ch("learn", "学習 ①", "挟む点を振って掴み、成否と特徴を記録する",
   ("gen_grasp_data.py", "sample"),
   "1つの学習データを作ります。形・見え方・向きを乱数で決めて部品を置き、真の形に対して挟む点を少しずつ振って掴み、成功したかを記録します。これを 8,000 回繰り返します。",
   io=["入力：データの番号 <code>i</code>（乱数の種は <code>20000 + i</code>）",
       "出力：1行の記録。特徴 <code>u, v, b, yaw, L, reach</code> と結果 <code>success</code>（ほかに確認用の値）"],
   points=["挟む点は、部品の軸に沿って手前の面から <code>a</code> ∈ [fl − 6mm, fl + L + 4mm]、横に <code>b</code> ∈ ±8mm で引きます。本体の外（フランジの上や奥の何もない所）も含めるので、失敗の例も十分に集まります。",
           "特徴の <code>u = a − fl</code> は本体の手前の縁から挟む点まで、<code>v = fl + L − a</code> は挟む点から本体の奥の縁までの距離です。",
           "<code>reach</code> は天板の前端から挟む点までの奥行きです。奥ほど手首が天板に近づくので、成否に関わります。",
           "掴む動作は <code>s.grasp()</code>（後の節）です。ここでは触らずに、天板の手前から差し込んで掴みます。"],
   note="実行：<code>python gen_grasp_data.py k n 8000</code>。n 並列のうち k 番目を担当し、結果は <code>results/gdata_k.jsonl</code> に1行ずつ追記されます。成功率は全体で 48% でした。")

ch("learn", "学習 ①", "掴む：差し込み、挟み、引き出して持ち上げる",
   ("shelf.py", "ShelfSim.grasp"),
   "挟む点 <code>target_xyz</code> に親指と中指の間を合わせて掴み、天板の外へ引き出して持ち上げます。学習データでも、推論の評価でも、この同じ関数で成否を決めます。",
   io=["入力：挟む点（世界座標）",
       "出力：<code>success</code>（4cm 以上持ち上がり、2秒保持できたか）、持ち上がった高さ、天板に当たった部位など"],
   points=["手の経路は学習していません。決めた経由点を逆運動学（mink）で解き、関節の目標を直線で補間する決めうちの動作です（<code>move_linear</code>）。",
           "触った後で手が天板の下にあるとき（<code>DIRECT</code>）は、腕を天板の外へ戻さず、今の高さ以上を保って挟む点の真上へ移ります。そうでなければ天板の手前から水平に差し込みます。",
           "差し込む高さは挟む点の 20mm 上（<code>PASS_H</code>）で、そこから真下に下ろしてから指を閉じます。",
           "成功の判定は <code>rise &gt; 0.04</code>：2秒保持した後に、最初の位置より 4cm 以上高いことです。"])

ch("learn", "学習 ①", "指を閉じる：先に触れた指は止め、両方触れたら締める",
   ("shelf.py", "close_each"),
   "親指と中指を、それぞれ開いた状態から閉じる方向へ進めます。部品に触れた指はそこで止め、もう一方だけを進めます。両方が触れたら、0.3秒かけて両方を同じだけ締めます。",
   io=["入力：シミュレーション、閉じる時間",
       "出力：各指が止まった位置。片方しか触れなければ <code>None</code>"],
   points=["同時に閉じると、先に当たった指が部品を押して動かします。指ごとに止めることで、挟む点がずれにくくなります。",
           "接触の判定は <code>d.contact</code> の中から、部品と指先のリンクの組を探しています。"])

ch("learn", "学習 ②", "成功確率の予測器を学習する",
   ("train_predictor.py", None, "FEAT = ", "pickle.dump"),
   "8,000件の記録から、特徴 6つ → 成功確率 を出す小さなニューラルネット（MLP）を学習し、<code>predictor.pkl</code> に保存します。これが図の f<sub>φ</sub> です。",
   io=["入力：<code>results/gdata_*.jsonl</code>",
       "出力：<code>predictor.pkl</code>（scikit-learn のモデル）と、較正の表"],
   points=["特徴は mm と度にそろえ、<code>StandardScaler</code> で標準化してから、隠れ層 64×64 の MLP に入れます。",
           "まず後ろの 2割（1,600件）を確認用に残して学習し、正解率（0.943）と Brier スコア（0.044）、予測の区間ごとの実際の成功率を出します。",
           "最後に全件で学習し直したものを保存します。",
           "特徴の並び <code>FEAT</code> は、推論側の <code>prob_matrix</code>（後の節）が作る列の並びと同じでなければなりません。"],
   note="実行：<code>python train_predictor.py</code>（数秒）。物理の値は特徴にないので、同じ特徴でも成功と失敗が混ざります。予測器はその割合、つまり物理のばらつきで平均した成功確率を学びます。")

# ===== 推論 ==================================================================================
ch("infer", "推論", "1配置の試行：①〜⑦ を呼ぶ順番",
   ("run_touchsel.py", "trial"),
   "主な比較（240配置 × 3つの選び方）の1試行です。図の ①〜⑦ の関数を、この順に呼んでいます。以下の節はこの順番で並べています。",
   io=["入力：配置の番号 <code>i</code>、触る点の選び方 <code>rule</code>（voi・unc・rand）、予測器のファイル",
       "出力：1回目・2回目に触った時点で掴んだ結果（<code>steps</code>）"],
   points=["配置 <code>RB.layout(i)</code>：見え方は <code>i % 4</code> で 10・15・20・25mm を順に、形・向き・物理の値は種 <code>1000 + i</code> の乱数で決めます。3つの選び方は同じ配置で比べます。",
           "① <code>P.observe</code> → ② <code>estimate_size</code>・<code>yaw_from_depth</code>・<code>init_particles_part</code>・<code>refine_part_depth</code> で、形の候補（粒子）200個を作ります。",
           "③ <code>choose</code> → ④ <code>s.touch</code> → ⑤ <code>F.update</code> → ⑥ <code>best_grasp_m</code> → ⑦ <code>branch_grasp</code> を、触る回数（2回）だけ繰り返します。",
           "掴む候補 <code>gc_</code> は軸に沿って 1mm おき、触る候補 <code>tc</code> は 3mm おきです。",
           "⑦ は状態を複製して掴むので、掴んだ後も元の状態から次の触りを続けられます。1つの試行から「1回触った時点」と「2回触った時点」の両方の成否が得られます。"],
   note="実行：<code>python run_touchsel.py k n 240 predictor.pkl</code> → <code>results/tsel_k.jsonl</code>。集計は <code>python analyze_touchsel.py</code>（1回触った時点の成功率は voi 95.4%、unc 88.8%、rand 88.8%）。")

ch("infer", "推論 ①", "頭部カメラで見る：領域分割と深度",
   ("perception.py", None, "MASK_NOISE = True", "return cam, dep, cube_mask"),
   "頭部カメラの画像を描画し、部品の領域（マスク）、机の領域、手前で隠すもの（天板・ロボット）の領域、深度画像を返します。",
   io=["入力：モデル <code>m</code>、状態 <code>d</code>、部品の名前",
       "出力：カメラ、深度画像、部品のマスク、机のマスク、遮蔽物のマスク"],
   points=["領域分割はシミュレーションの正解（どの形状が写っているか）を使い、実機の領域分割モデルの誤りを <code>noisy_mask</code> でまねます。境界を ±1 画素ずらし、境界の画素を 30% の確率で反転します。",
           "深度には D435i の雑音モデル σ(z) = z²·0.08 / (f·b) を足します（<code>HeadCam.render</code>）。",
           "遮蔽物のマスク <code>occl</code> は「部品でも机でもないもの」です。次の照合で、そこに隠れた部分は見えなくて当然として扱います。"])

ch("infer", "推論 ②", "机の高さを測る",
   ("perception.py", "estimate_size"),
   "深度画像を世界座標の点群にして、部品のまわり 10cm の机の点から机の高さ <code>ztab</code> を出します。試行で使うのはこの机の高さです。",
   io=["入力：カメラ、深度、部品と机のマスク", "出力：(部品の高さ, 机の高さ)"],
   points=["机の高さは中央値で出すので、境界の誤りで混じった点に引きずられません。",
           "部品の高さ（見えている点の上位 5% の中央値 − 机の高さ）は、立方体を扱っていた頃の名残です。"])

ch("infer", "推論 ②", "部品の向きを深度から出す",
   ("perception.py", None, "YAW_SD = ", "return float(-np.arctan(t))"),
   "フランジの手前の縁を深度の点で直線に当てはめ、その傾きから部品の向き（yaw）を出します。",
   io=["入力：カメラ、深度、部品のマスク、机の高さ", "出力：向き [rad]。点が足りなければ <code>None</code>"],
   points=["高さ 1.5〜7mm の点だけを使います。フランジの上面と手前の面で、本体（20mm）の点は混ぜません。",
           "左右方向に 2mm 幅の帯に分け、各帯で最も手前（x の 5% 点）を縁の位置とします。その並びに直線を当てはめ、傾き t から yaw = −atan(t) です。",
           "領域の形だけで向きを推定すると事前分布の中央（0°）に寄ってしまったので、この方法にしました（コメントに経緯があります）。",
           "<code>YAW_SD</code>（3.5°）は、40配置で測ったこの推定の誤差から決めた値で、次の節で候補の向きのばらつきに使います。"])

ch("infer", "推論 ②", "形の候補（粒子）を 200個つくる",
   ("perception.py", "init_particles_part"),
   "位置・向き・フランジの長さ・本体の長さの組を 3,000個引き、それぞれ「その形ならカメラに見えるはずの領域」を描いて観測のマスクと照合し、よく合うものほど多く残るように 200個を選び直します。",
   io=["入力：カメラ、部品と遮蔽物のマスク、机の高さ、事前分布の中心、長さの範囲、深度から出した向き",
       "出力：粒子 <code>parts</code>（200×5：手前の面の x、中心の y、yaw、フランジの長さ fl、本体の長さ L）"],
   points=["候補の引き方：手前の面の x は事前の中心から ±30mm（標準偏差）、y は ±15mm、向きは深度の推定 ± 3.5°、fl は 10〜30mm、L は 12〜20mm の一様分布です。",
           "見えるはずの領域：フランジ（高さ 4mm）と本体（高さ 20mm）の箱の8頂点をカメラに投影し、凸包を塗ります。そこから遮蔽物の画素を除きます。",
           "照合：予測の境界 1画素は数えません（縮めた予測 <code>p_in</code> と広げた予測 <code>p_out</code> を使う）。境界の誤りを数えると「本体が隠れている」側に偏ったためです。",
           "重みは exp(−(不一致率 − 最小) / TEMP) です。TEMP は <code>run_branch.py</code> で 0.05 に設定しています。"],
   note="カメラにはフランジの手前しか見えないことが多いので、この段階では fl と L はほとんど決まりません。決まらない分を触って確かめる、というのがこの研究の出発点です。")

ch("infer", "推論 ②", "本体が見えているかで、フランジの長さを引き直す",
   ("perception.py", "refine_part_depth"),
   "深度で高さ 10mm 以上の点（＝本体）が見えているかを調べ、粒子のフランジの長さ fl を引き直します。",
   io=["入力：粒子、カメラ、深度、部品のマスク、机の高さ、天板の前端と下面の高さ",
       "出力：(更新した粒子, 本体が見えたか)"],
   points=["見えている場合：粒子ごとに、見えた本体の点の軸方向の位置を出し、その 30% 点を本体の手前の縁とします。fl はそこから ±1.5mm で引きます。",
           "見えていない場合：高さ 10mm の点が天板の縁に隠れ始める位置 <code>x_hide</code> を出し、本体の手前の縁はそれより奥にあるとして、fl を「そこから上限まで」の一様分布で引き直します。",
           "領域の形だけでは、本体を奥にずらしても見える形がほとんど変わりませんでした。深度の高さの情報で補っています。"])

ch("infer", "推論 ③⑥", "候補点：推定した部品の軸に沿って並べる",
   ("partfilter.py", "axis_points"),
   "粒子の平均の手前の面・y・向きから部品の軸を1本決め、その上に候補点を並べます。掴む点の候補（1mm おき）と触る点の候補（3mm おき）の両方に使います。",
   io=["入力：粒子、間隔 <code>step</code>、範囲（手前の面から 0〜55mm）", "出力：候補点 (x, y) のリスト"],
   points=["向きの平均は角度をそのまま平均せず、sin と cos を平均してから atan2 で戻します。"])

ch("infer", "推論 ③⑥", "粒子 × 挟む点 の成功確率の行列を、予測器で出す",
   ("partfilter.py", None, "def use_predictor", "return _PRED.predict_proba"),
   "学習した予測器 f<sub>φ</sub> を読み込み、粒子 200個 × 挟む点の候補 の全組について成功確率を出します。③と⑥はどちらもこの行列を使います。",
   io=["入力：粒子（N×5）、挟む点の候補（G個）", "出力：成功確率の行列（N×G）"],
   points=["各粒子の座標系で、挟む点の軸方向の位置 a と横のずれ b を出し、学習と同じ6つの特徴 u = a − fl、v = fl + L − a、b、yaw、L、reach を作ります。単位も学習と同じ mm と度です。",
           "N×G 個の特徴をまとめて1回で <code>predict_proba</code> に渡すので、速く計算できます。",
           "予測器を読み込まなければ、本体の縁から内側に余裕を持った範囲なら成功とみなす幾何の規則を使います（予測器を作る前の版）。"])

ch("infer", "推論 ⑥", "掴む点を選ぶ：候補で平均した成功確率が最大の点",
   ("partfilter.py", "best_grasp_m"),
   "行列を粒子の方向に平均すると、挟む点ごとの「形の不確かさを込みにした成功確率」になります。その最大の点を掴む点とします。",
   io=["入力：粒子、挟む点の候補", "出力：(掴む点, その成功確率)"],
   points=["1つの形を決めてから掴む点を選ぶのではなく、候補の形すべてで平均してから選びます。どの形でもそこそこ掴める点が選ばれます。"])

ch("infer", "推論 ③", "触る点を選ぶ（本手法 voi）：触った後の最良の成功確率の期待値が最大の点",
   ("partfilter.py", "choose_touch_m"),
   "触る点の候補ごとに、「そこを触ったら何が観測されそうか」と「観測ごとに更新した後、最良の掴む点の成功確率はいくつか」を計算し、その期待値が最大の点を選びます。この研究の中心の式です。",
   io=["入力：粒子、掴む点の候補、触る点の候補", "出力：(触る点, 期待値)"],
   points=["式で書くと u* = argmax<sub>u</sub> Σ<sub>o</sub> P(o | u, b) · max<sub>g</sub> P(成功 | g, b<sub>u,o</sub>) です。",
           "<code>lk</code> は観測 o が得られる確率を粒子ごとに並べたもので、その平均 <code>s / len(parts)</code> が P(o | u, b) です。",
           "<code>lk / s</code> は観測 o の後の粒子の重みです。<code>(lk / s) @ Pm</code> で、更新後の各挟む点の成功確率が一度に出ます。その最大が max<sub>g</sub> の項です。",
           "成功確率の行列 <code>Pm</code> は最初に1回だけ計算し、使い回します。更新は粒子の重みを変えるだけなので、行列は同じものが使えます。",
           "形を正確に知るための触り方ではなく、「掴む点の選び方が最も変わる」触り方を選ぶ点が、比べる2つの規則との違いです。"])

ch("infer", "推論 ③", "比べる規則：形の不確かさ最大（unc）とランダム（rand）",
   ("run_touchsel.py", None, "def occupancy", "return tc[rng.integers"),
   "比較のための触る点の選び方です。<code>rule</code> で切り替え、ほかの手順はすべて同じです。",
   io=["入力：規則の名前、粒子、候補", "出力：触る点"],
   points=["unc：点の真下に本体があると考える粒子の割合 p を出し、p(1 − p) が最大の点を触ります。p が 0.5 に近い、つまり本体があるかないかが最も分からない場所です。Act-VH の「形の不確かさが最大の所を触る」考え方を、この問題に当てはめたものです。",
           "rand：触る点の候補から一様に選びます。",
           "voi は前の節の <code>choose_touch_m</code> を呼ぶだけです。"])

ch("infer", "推論 ④", "中指で上から軽く触る",
   ("shelf.py", "ShelfSim.touch"),
   "中指の先を触る点の真上 10mm から 1mm ずつ下ろし、接触力が 0.1N（約 10g）を超えたら止めて、何に当たったかを4通りに分けて返します。",
   io=["入力：触る点 (x, y)",
       "出力：観測 <code>obs</code>（0 外れ／1 指先の中央で本体／2 指の腹など中央以外で本体／3 フランジ）と、部品が動いた量など"],
   points=["手首の向きは掴むときと同じ（−55°）なので、触った後に持ち替えずに掴みに移れます。",
           "親指は中指より約 5mm 低いので、触る間だけ上へ退避させます（<code>THUMB_UP</code>）。",
           "机の高さまで下ろしても部品に当たらなければ外れです。机や天板に当たった場合も外れ（0）になります。",
           "当たった点が中指の先の中央から 4mm 以内なら 1、それ以外なら 2、当たった形状がフランジなら 3 です。",
           "最後に指を触り始めの高さへ戻します。<code>DIRECT</code> のときは天板の下にいたまま、次の触る点や掴む点へ移ります。"])

ch("infer", "推論 ⑤", "触る観測のモデル：形の候補ごとに、各観測が起こる確率",
   ("partfilter.py", None, "MISS, CENTER, OTHER, FLANGE", "return EPS / 4"),
   "粒子ごとに、「この形のとき、この点を触ると観測 o が出る確率」を返します。③（期待値の計算）と⑤（更新）の両方で使います。",
   io=["入力：粒子、触る点、観測 o", "出力：粒子ごとの確率（長さ 200）"],
   points=["本体の手前の縁からの距離 u と、奥の縁からの距離 v で場合分けします。u &lt; −11mm はフランジ、−11〜−3mm は指の腹で本体、−3mm から v = +2mm までは指先の中央で本体、v = +2〜+12mm は指の腹で本体、それより奥は外れです。",
           "境界の値は予備実験で測ったものです。指の形のせいで、本体の縁より手前や奥を触っても本体に当たる幅があります。",
           "境界は幅 1mm（<code>BLUR</code>）のシグモイドでぼかしています。位置の誤差と指の形のばらつきを見込んだものです。",
           "どの観測にも 2%（<code>EPS</code>）の確率を残します。まれな外れ値で全部の粒子の確率が 0 になり、推定が壊れるのを防ぎます。"])

ch("infer", "推論 ⑤", "触った結果で候補を更新する",
   ("partfilter.py", "update"),
   "実際に得た観測の確率を重みにして、粒子 200個を選び直します（パーティクルフィルタの更新）。",
   io=["入力：粒子、触った点、観測、乱数", "出力：更新した粒子"],
   points=["重みの大きい粒子ほど多く複製されます。観測と合わない形の候補は消えていきます。",
           "選び直した後に小さなずれ（位置と長さ 0.3mm、向き 0.5°）を足し、同じ粒子が重なりすぎないようにします。"])

ch("infer", "推論 ⑦", "状態を複製して掴む（評価のための掴み）",
   ("run_branch.py", "branch_grasp"),
   "今のシミュレーションの状態 <code>MjData</code> を複製し、複製の上で掴んで成否を記録します。元の状態には触れないので、掴んだ後も触る前の続きから試行を進められます。",
   io=["入力：シミュレーション、掴む点", "出力：成否、失敗のときの部品の移動量など"],
   points=["1つの試行の中で「k 回触った時点で掴んだらどうなったか」を k ごとに測れます。同じ観測の続きなので、触る回数の比較を公平にできます。",
           "<code>mj_setState</code> で状態を戻す方法も試しましたが、戻した直後の掴みが不安定になったので、複製する方法にしました。",
           "実機ではこの複製はできません。評価のためだけの仕組みで、手法の一部ではありません。"])

# ===== SmolVLM ===============================================================================
ch("vlm", "SmolVLM", "全体の流れ：画像と指示文で部品を選び、選んだ部品を触って掴む",
   ("vlm_pick.py", None, 'a = ap.parse_args()', "pick(a.layout"),
   "色と隠れたサイズの違う部品を2つ並べ、SmolVLM に「どちらを取るか」を番号で答えさせ、選ばれた部品を推論の ①〜⑦ と同じ手順で触って掴みます。",
   io=["入力：指示文（例 \"Pick up the green part.\"）、色の並び、部品ごとのサイズ",
       "出力：SmolVLM に渡した画像 <code>vlm_input.png</code>、問いと答え <code>vlm_answer.json</code>、掴んだ結果（動画も保存できる）"],
   points=["サイズが予測器を学習した範囲の外なら、<code>assert</code> で止めます。範囲の外では成功確率の予測が当てになりません。",
           "SmolVLM に見せる画像は、選ぶ前の場面から作ります。天板の前端が共通なので、どちらの部品もフランジの手前が同じだけ見え、サイズの違いは見えません。",
           "<code>--answer</code> を与えると SmolVLM を使わずに答えを手で与えられます。GPU のない環境で手順だけ確かめるときに使います。",
           "分担は SayCan と同じ骨組みです。言葉と画像から「何を取るか」を SmolVLM が、「どこを触ってどこを挟むか」を物理から学んだ成功確率が決めます。"])

ch("vlm", "SmolVLM", "部品を2つ並べた場面をつくる",
   ("vlm_pick.py", None, "def add_extras", "return s, lay"),
   "取る部品（動く剛体）を選んだ位置に置き、もう一方の位置には、色とサイズの違う部品を動かない置物として置きます。",
   io=["入力：配置の番号、取る位置、色、サイズ", "出力：シミュレーションと配置の値"],
   points=["位置は今の位置と、そこから右へ 10cm です。左へずらすと手が届かず、5cm 間隔では隣の部品に手が当たって掴めませんでした。",
           "置物には関節を付けないので、腕が当たっても動きません。触る・掴むのは取る部品だけです。",
           "<code>bs.build_shelf</code> を一時的に差し替えて置物を足し、作り終えたら元に戻しています。"])

ch("vlm", "SmolVLM", "SmolVLM に見せる画像：切り出して拡大し、番号を書く",
   ("vlm_pick.py", "vlm_image"),
   "頭部カメラの画像（640×480）から2つの部品のまわりを切り出して 3倍に拡大し、それぞれの上に番号 1・2 を書きます。",
   io=["入力：シミュレーション、配置", "出力：PIL の画像"],
   points=["元の画像のままでは部品が小さく、SmolVLM が色を見分けにくいので拡大しています。",
           "切り出す範囲は画像の外にはみ出さないように切り詰めます。はみ出すと拡大のときに画像が引き伸ばされます。"])

ch("vlm", "SmolVLM", "SmolVLM に問い、番号を読み取る",
   ("vlm_pick.py", None, "def ask_smolvlm", "return None if m is None"),
   "画像と問いの文をチャットの形式で SmolVLM に渡し、短い答えを生成させて、その中から番号 1 か 2 を読み取ります。",
   io=["入力：画像、指示文、モデル名（既定 SmolVLM-500M-Instruct）", "出力：問い、答えの文字列、使った装置"],
   points=["問いの文 <code>QUESTION</code> は「番号だけで答えよ」と指示しています。生成は8トークンまで、乱数を使わない（<code>do_sample=False</code>）ので、同じ入力なら同じ答えになります。",
           "GPU があれば bfloat16 で GPU に載せます。",
           "transformers の版の違い（v5 で <code>AutoModelForVision2Seq</code> がなくなり、<code>torch_dtype</code> が <code>dtype</code> になった）を吸収しています。"])

ch("vlm", "SmolVLM", "選ばれた部品を触って掴む",
   ("vlm_pick.py", "pick", None, 'g = s.grasp([gx, gy'),
   "推論の ①〜⑦ と同じ関数を同じ順に呼びます。違うのは、形の候補を作るときの事前分布の中心を、選ばれた部品の位置へずらす所だけです。",
   io=["入力：配置、選ばれた位置、色、サイズ、触る回数（既定 1）", "出力：掴めたか"],
   points=["<code>prior = (S.X_FRONT, S.Y0 + SLOTS[slot])</code> が事前分布の中心です。",
           "触る点は voi（<code>T.choose(\"voi\", ...)</code>）で選びます。",
           "<code>--touches 0</code> にすると触らずに、カメラの推定だけで掴みます。比べるときに使います（<code>run_vlm_compare.py</code> は同じ比較を 120回まとめて行います）。"])

# ---------------------------------------------------------------------------------------------
ORDER = [
    ("学習", "1", "python gen_grasp_data.py k n 8000", "把持を 8,000回試して、特徴と成否を記録する", "→ results/gdata_*.jsonl", "learn"),
    ("学習", "2", "python train_predictor.py", "成功確率の予測器 f<sub>φ</sub> を学習する", "results/gdata_*.jsonl → predictor.pkl", "learn"),
    ("推論", "3", "python run_touchsel.py k n 240 predictor.pkl", "主な比較：240配置 × 触る点の選び方3通り", "predictor.pkl → results/tsel_*.jsonl", "infer"),
    ("推論", "4", "python run_branch.py k n 120 predictor.pkl", "触る回数の比較：120配置で 0〜4回", "predictor.pkl → results/branch_*.jsonl", "infer"),
    ("集計", "5", "python analyze_touchsel.py<br>python analyze_branch.py", "成功率、信頼区間、McNemar の検定", "results/*.jsonl → 画面に表", None),
    ("組み合わせ", "6", "python vlm_pick.py --instruct \"Pick up the green part.\"", "SmolVLM で選び、触って掴む（1回のデモ）", "predictor.pkl → vlm_input.png・vlm_answer.json", "vlm"),
    ("見る", "—", "python view_trial.py 64 --rule voi", "1配置の試行を表示ウィンドウで見る・動画に保存する", "predictor.pkl → 表示・mp4", None),
]

OTHERS = [
    ("grasp_test.py", "手の開閉の形、手首の向き、関節空間での腕の移動など、把持の土台（最初の把持の試験から育ったもの）"),
    ("ik_mink.py", "逆運動学を mink に差し替える。右腕7関節だけを動かし、ほかは固定"),
    ("check_scene.py", "シーンが組み上がるかの確認"),
    ("analyze_touchsel.py・analyze_branch.py", "結果の集計。成功率、ブートストラップの 95% 区間、McNemar の正確検定"),
    ("view_trial.py", "1配置の試行を表示ウィンドウで見る・動画に保存する（--close・--macro で寄りの映像）"),
    ("run_vlm_compare.py", "SmolVLM で選んだ部品を、触らずに掴む場合と1回触ってから掴む場合で 120回比べる"),
    ("run_vlm_point.py", "掴む点まで SmolVLM に選ばせる比較（候補点 A〜F）"),
    ("tools/*.py", "レポートと動画の図（fig_*.py）、提出動画（make_video.py）、この解説（make_walkthrough.py）"),
]

STEPS = [
    ("準備：uv と git を入れる",
     ["# Windows（PowerShell）\npowershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"",
      "# Mac・Linux\ncurl -LsSf https://astral.sh/uv/install.sh | sh"],
     "Python 3.11 は uv が自動で入れます。"),
    ("リポジトリを取ってきて、依存を入れる",
     ["git clone https://github.com/Nishina-N/Physics-Informed-Touch-Selection.git\ncd Physics-Informed-Touch-Selection\nuv sync"],
     "MuJoCo 3.14、mink、scikit-learn 1.8.0 などが <code>.venv</code> に入ります。"),
    ("ロボットのモデル（mujoco_menagerie）を置く",
     ["git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie\n"
      "git -C third_party/mujoco_menagerie checkout c96a32d28fb5da84da38c1da4d749e7a13212855"],
     "レポートと同じ版に固定します。動画の字幕に日本語のフォントを使うので、Ubuntu・WSL では <code>sudo apt install fonts-noto-cjk</code> も入れます。"),
    ("保存済みの結果から数値を出す",
     ["uv run python analyze_touchsel.py\nuv run python analyze_branch.py"],
     "期待される出力：<code>voi 成功率 0.954</code>、<code>unc 成功率 0.887</code>、<code>voi − unc: +0.067 … McNemar p=0.005</code>、<code>A 0.725</code>、<code>B1 0.967</code>。"),
    ("予測器を学習し直す",
     ["uv run python train_predictor.py"],
     "期待される出力：<code>Brier 0.044 正解率 0.943</code>。<code>predictor.pkl</code> は上書きされます（このリポジトリの環境では元と同じファイルになりました。戻すときは <code>git checkout predictor.pkl</code>）。"),
    ("シミュレーションを1配置だけ動かす",
     ["uv run python -c \"import run_branch as B; r = B.trial(0, 'predictor.pkl'); "
      "print([(s['k'], s['success'], round(s['p'], 3)) for s in r['steps']])\""],
     "期待される出力：<code>[(0, True, 0.578), (1, True, 0.861), (2, True, 0.952), (3, True, 0.975), (4, True, 0.991)]</code>。"
     "触るたびに、予測した成功確率 p が上がっていきます。保存済みの <code>results/branch_0.jsonl</code> の配置 0 と同じになります。"),
    ("ロボットの動きを見る",
     ["uv run python view_trial.py 64 --rule voi\nuv run python view_trial.py 64 --rule unc",
      "# Mac は表示ウィンドウに mjpython が要る\nuv run mjpython view_trial.py 64 --rule voi",
      "# 動画に保存（表示ウィンドウは開かない）\nuv run python view_trial.py 64 --rule voi --video trial64_voi.mp4"],
     "配置 64（見え方 10mm、本体は手前の面から 15〜31mm）では、voi は奥の 43mm を触って外れ、「本体はそこまで奥にない」と分かって掴めます。"
     "unc は 23mm を触って本体に当たりますが、本体の手前の縁のばらつきが 4.3mm 残り、掴み損ねます。"),
    ("SmolVLM と組み合わせる（任意）",
     ["uv pip install transformers torch pillow   # GPU を使うなら CUDA 版の torch\n"
      "uv run python vlm_pick.py --instruct \"Pick up the green part.\"",
      "# SmolVLM なしで手順だけ試す\nuv run python vlm_pick.py --answer 2 --video test.mp4"],
     "初回は Hugging Face からモデル（約 1GB）を取ってきます。"),
    ("実験を全部やり直す（任意）",
     ["# 保存済みの結果を退避（残すと、済んだ配置を飛ばして何もしない）\nmv results results_paper\n"
      "uv run python gen_grasp_data.py 0 2 8000\nuv run python train_predictor.py\n"
      "uv run python run_touchsel.py 0 2 240 predictor.pkl\nuv run python run_branch.py 0 2 120 predictor.pkl"],
     "<code>0 2</code> は「2並列のうち 0番目」です。別の端末で <code>1 2</code> も同時に走らせます。途中で止めても、同じコマンドで続きから再開します。"
     "目安は、学習データが 1並列で約 3.5時間、<code>run_branch</code> の 120配置が約 26分です。"),
]

PARTS = [("learn", "学習", "事前に1回。物理シミュレーションで把持を試し、成功確率の予測器 f<sub>φ</sub> を作る。"),
         ("infer", "推論（1配置の試行）", "配置ごとに、見て、触って、候補を更新し、掴む。図の ①〜⑦ の順に並べている。"),
         ("vlm", "SmolVLM との組み合わせ", "指示文と画像から取る部品を選ぶ部分を足したデモ。触る・掴むは推論と同じ関数を使う。")]

CSS = """
:root{--bg:#f6f6f3;--panel:#fff;--code:#fafaf7;--line:#e2e3dd;--fg:#1d2120;--muted:#5f6763;--accent:#1e5faa;--soft:#e8f0f9;
--hi:#fff6e0;--k:#8a3b7a;--s:#2e6b47;--n:#a0522d;--c:#7d8681;--f:#1e5faa;--b:#9a6a1c;
--sans:"Hiragino Sans","Noto Sans JP","Yu Gothic UI","Yu Gothic",system-ui,sans-serif;
--mono:ui-monospace,"SFMono-Regular",Menlo,Consolas,"Noto Sans Mono CJK JP","BIZ UDGothic",monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121514;--panel:#191d1b;--code:#151917;--line:#2b322f;
--fg:#e3e8e5;--muted:#9aa49f;--accent:#8ab8e0;--soft:#1c2a36;--hi:#2c2718;--k:#d49ac6;--s:#8fcaa4;--n:#e0a37a;--c:#7f8a85;--f:#8ab8e0;--b:#d9b36b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121514;--panel:#191d1b;--code:#151917;--line:#2b322f;--fg:#e3e8e5;--muted:#9aa49f;--accent:#8ab8e0;
--soft:#1c2a36;--hi:#2c2718;--k:#d49ac6;--s:#8fcaa4;--n:#e0a37a;--c:#7f8a85;--f:#8ab8e0;--b:#d9b36b;color-scheme:dark}
*{box-sizing:border-box}html{scroll-padding-top:64px}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.75}
a{color:var(--accent)}code{font-family:var(--mono);font-size:.86em;background:var(--code);border:1px solid var(--line);border-radius:4px;padding:0 .3em}
.wrap{max-width:1440px;margin:0 auto;padding:24px 16px 80px}
nav.top{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(6px);border-bottom:1px solid var(--line)}
nav.top .in{max-width:1440px;margin:0 auto;padding:8px 16px;display:flex;gap:6px;flex-wrap:wrap;align-items:center}
nav.top a{text-decoration:none;color:var(--fg);font-size:.85rem;padding:3px 10px;border:1px solid var(--line);border-radius:999px;background:var(--panel)}
nav.top a:hover{border-color:var(--accent);color:var(--accent)}nav.top b{font-size:.85rem;margin-right:6px}
h1{font-size:1.7rem;line-height:1.35;margin:18px 0 6px}h2{font-size:1.3rem;margin:0}h3{font-size:1.02rem;margin:0;line-height:1.45}
.lead{color:var(--muted);max-width:80ch;margin:0}
section{margin-top:44px}section>.sh{display:grid;gap:4px;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid var(--fg)}
section>.sh p{color:var(--muted);margin:0}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px}figure img{width:100%;height:auto;display:block}
figcaption{color:var(--muted);font-size:.85rem;padding:6px 4px 0}
.tbl{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}
table.t{border-collapse:collapse;width:100%;font-size:.9rem}table.t th,table.t td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
table.t th{background:var(--code);font-weight:600;white-space:nowrap}table.t tr:last-child td{border-bottom:0}
table.t td.cmd{font-family:var(--mono);font-size:.82rem;white-space:nowrap}
.flow{display:grid;gap:16px}.flow figure{max-width:980px}
article.ch{margin-top:22px;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
article.ch>header{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid var(--line);background:var(--code)}
.tag{font-size:.78rem;font-weight:700;color:var(--accent);background:var(--soft);border-radius:4px;padding:1px 8px;white-space:nowrap}
.loc{margin-left:auto;font-family:var(--mono);font-size:.8rem;color:var(--muted);text-decoration:none}.loc:hover{color:var(--accent)}
.body{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr)}
@media (max-width:980px){.body{grid-template-columns:minmax(0,1fr)}}
.src{border-right:1px solid var(--line);background:var(--code);overflow-x:auto;min-width:0}
@media (max-width:980px){.src{border-right:0;border-bottom:1px solid var(--line);max-height:70vh;overflow:auto}}
.ex{padding:14px 18px;min-width:0}.ex .stick{position:sticky;top:60px;display:grid;gap:10px}
.ex p{margin:0}.lbl{font-size:.78rem;font-weight:700;color:var(--muted);letter-spacing:.04em;margin-bottom:2px}
.ex ul,.ex ol{margin:0;padding-left:1.3em;display:grid;gap:4px}.note{color:var(--muted);font-size:.88rem;border-left:3px solid var(--line);padding-left:10px}
.hltable{border-collapse:collapse;font-family:var(--mono);font-size:12.5px;line-height:1.6}
.hltable td{padding:0;vertical-align:top}.hltable pre{margin:0;font-family:inherit}
.linenos{user-select:none;color:var(--c);text-align:right;padding:10px 10px 10px 12px!important;border-right:1px solid var(--line)}
.hltable td.code{padding:10px 14px!important}.hl{background:none}.hl code{border:0;background:none;padding:0;font-size:inherit;border-radius:0}
@media (max-width:980px){html{scroll-padding-top:110px}}
.hl .k,.hl .kn,.hl .kd,.hl .ow,.hl .kc{color:var(--k);font-weight:600}.hl .s,.hl .s1,.hl .s2,.hl .sa,.hl .se,.hl .si{color:var(--s)}
.hl .sd{color:var(--c)}.hl .c,.hl .c1,.hl .cm{color:var(--c);font-style:italic}.hl .mi,.hl .mf,.hl .il{color:var(--n)}
.hl .nf,.hl .nc,.hl .fm{color:var(--f);font-weight:600}.hl .nb,.hl .bp{color:var(--b)}
.steps{display:grid;gap:12px}.step{display:grid;grid-template-columns:28px minmax(0,1fr);gap:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.step .no{width:26px;height:26px;border-radius:50%;background:var(--soft);color:var(--accent);font-weight:700;display:grid;place-items:center;font-size:.85rem}
.step .b{display:grid;gap:6px;min-width:0}
.cmd{position:relative;font-family:var(--mono);font-size:.8rem;background:var(--code);border:1px solid var(--line);border-radius:6px;padding:8px 64px 8px 10px;white-space:pre;overflow-x:auto}
.copy{position:absolute;top:5px;right:5px;font:inherit;font-size:.72rem;padding:2px 8px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--fg);cursor:pointer}
.copy:hover{border-color:var(--accent)}.step .ex2{color:var(--muted);font-size:.9rem;margin:0}
footer{margin-top:48px;color:var(--muted);font-size:.85rem}
"""

JS = """
document.querySelectorAll('.copy').forEach(b=>b.addEventListener('click',()=>{
 const t=b.parentElement.innerText.replace(/^(コピー|コピーしました)\\n?/,'').replace(/(コピー|コピーしました)$/,'');
 const done=()=>{b.textContent='コピーしました';setTimeout(()=>b.textContent='コピー',1500)};
 if(navigator.clipboard)navigator.clipboard.writeText(b.parentElement.dataset.cmd).then(done).catch(()=>{});
}));
"""


def esc(s):
    return html.escape(s, quote=True)


def chunk_html(c, commit, idx):
    file, qual, start, end = (list(c["src"]) + [None, None, None])[:4]
    a, b = find(file, qual, start, end)
    link = f"{REPO}/blob/{commit}/{file}#L{a}-L{b}"
    name = f"{file}" + (f" : {qual}" if qual else "") + f"（{a}–{b}行）"
    ex = [f"<p>{c['role']}</p>"]
    if c["io"]:
        ex.append("<div><div class='lbl'>入力と出力</div><ul>" + "".join(f"<li>{x}</li>" for x in c["io"]) + "</ul></div>")
    if c["points"]:
        ex.append("<div><div class='lbl'>読みどころ</div><ol>" + "".join(f"<li>{x}</li>" for x in c["points"]) + "</ol></div>")
    if c["note"]:
        ex.append(f"<p class='note'>{c['note']}</p>")
    return (f"<article class='ch' id='c{idx}'><header><span class='tag'>{c['no']}</span><h3>{c['title']}</h3>"
            f"<a class='loc' href='{link}' target='_blank' rel='noopener'>{esc(name)} ↗</a></header>"
            f"<div class='body'><div class='src'>{code_html(file, a, b)}</div>"
            f"<div class='ex'><div class='stick'>{''.join(ex)}</div></div></div></article>")


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    nlines = 0
    body = []
    # 全体の流れ
    first = {p: next(i for i, c in enumerate(CH) if c["part"] == p) for p, _, _ in PARTS}
    rows = "".join(
        f"<tr><td>{g}</td><td>{n}</td><td class='cmd'>{cmd}</td><td>{what}</td><td>{io}</td>"
        f"<td>{'' if p is None else f'<a href=#c{first[p]}>解説へ</a>'}</td></tr>"
        for g, n, cmd, what, io, p in ORDER)
    body.append(f"""<section id="flow"><div class="sh"><h2>全体の流れ</h2>
<p>学習は事前に1回、推論は配置ごとに行います。下の表の順にスクリプトを実行し、各節では左にそのコード、右に解説を置いています。</p></div>
<div class="flow"><figure><img src="figures/fig0_pipeline.png" alt="学習と推論の流れのブロック図" loading="lazy">
<figcaption>学習と推論の流れ（レポートの図）。①〜⑦ は下の「推論」の節の番号と同じです。</figcaption></figure>
<div class="tbl"><table class="t"><tr><th>段</th><th>順</th><th>実行するコマンド</th><th>何をするか</th><th>読む → 書く</th><th></th></tr>{rows}</table></div></div>
<p class="note" style="margin-top:12px">ファイルの役割：<code>build_scene.py</code> がシーンを組み立て、<code>shelf.py</code> の <code>ShelfSim</code> が触る・掴む動作を行い、
<code>perception.py</code> がカメラから形の候補を作り、<code>partfilter.py</code> が候補の更新と、成功確率・触る点・掴む点の計算を受け持ちます。
<code>gen_grasp_data.py</code>・<code>run_*.py</code> はそれらを呼ぶ実験の入口です。</p></section>""")
    for p, title, lead in PARTS:
        arts = []
        for i, c in enumerate(CH):
            if c["part"] == p:
                arts.append(chunk_html(c, commit, i))
                file, qual, start, end = (list(c["src"]) + [None, None, None])[:4]
                a, b = find(file, qual, start, end); nlines += b - a + 1
        body.append(f"<section id='{p}'><div class='sh'><h2>{title}</h2><p>{lead}</p></div>{''.join(arts)}</section>")
    steps = "".join(
        f"<div class='step'><span class='no'>{i + 1}</span><div class='b'><h3>{h}</h3>"
        + "".join(f"<div class='cmd' data-cmd='{esc(cmd)}'><button class='copy' type='button'>コピー</button>{esc(cmd)}</div>" for cmd in cmds)
        + f"<p class='ex2'>{ex}</p></div></div>" for i, (h, cmds, ex) in enumerate(STEPS))
    body.append(f"<section id='run'><div class='sh'><h2>手元で動かす</h2><p>リポジトリの直下で実行します。結果は <code>results/</code> に保存済みなので、手順4までは数分で終わります。</p></div><div class='steps'>{steps}</div></section>")
    others = "".join(f"<tr><td><code>{f}</code></td><td>{t}</td></tr>" for f, t in OTHERS)
    body.append(f"<section id='others'><div class='sh'><h2>ほかのファイル</h2><p>上の節に出てこないもの。</p></div><div class='tbl'><table class='t'>{others}</table></div></section>")
    nav = "".join(f"<a href='#{k}'>{t}</a>" for k, t in [("flow", "全体の流れ"), ("learn", "学習"), ("infer", "推論"),
                                                        ("vlm", "SmolVLM"), ("run", "手元で動かす"), ("others", "ほかのファイル")])
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>触る点の選択 コードの読み方</title><style>{CSS}</style></head><body>
<nav class="top"><div class="in"><b>コードの読み方</b>{nav}</div></nav>
<div class="wrap"><header><h1>棚の奥で一部が隠れた部品を掴むための、触る点の選択：コードの読み方</h1>
<p class="lead">学習と推論で、どのスクリプトがどの順に何をしているかを、実行する順に追えるようにしたものです。左に要の関数の抜粋（{len(CH)}か所、計{nlines}行）、右に解説を置いています。
コードは <a href="{REPO}">GitHub のリポジトリ</a>のコミット <code>{commit[:7]}</code> のもので、見出しの右のリンクから元のファイルの該当行へ移れます。</p></header>
{''.join(body)}
<footer>この頁は <code>tools/make_walkthrough.py</code> で作っています。コードを直したら <code>uv run python tools/make_walkthrough.py</code> で作り直してください。</footer>
</div><script>{JS}</script></body></html>"""
    open(OUT, "w", encoding="utf-8").write(page)
    print(f"{OUT}（{len(CH)}節、{nlines}行）")


if __name__ == "__main__":
    main()
