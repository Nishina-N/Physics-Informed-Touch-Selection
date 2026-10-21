# 棚の奥で一部が隠れた部品を掴むための触る点の選択

Physical AI 講座 最終課題のコード・学習済みモデル・実験結果です。
MuJoCo 上の Unitree G1＋Dex3（3指ハンド）が、上の段の天板で奥が隠れた小部品を、
見えた部分と軽く触った結果から形を推定して掴みます。

## 概要

- 部品は手前に低いフランジ（高さ4mm）、奥に掴める本体（幅・高さ20mm）をもつ。フランジの長さ ℓ_f（10〜30mm）と本体の長さ ℓ_b（12〜20mm）は未知。
- 頭部カメラの領域分割と深度から、形の候補 θ = (位置, 向き, ℓ_f, ℓ_b) を 200 個の粒子で推定する。
- 把持成功確率は、候補ごとに小さな MLP（物理シミュレーションの 8,000 回の把持から学習）で出し、候補で平均する。
- **触る点は「触った後に得られる最良の成功確率の期待値」が最大の点**として選ぶ（`voi`）。
  比較として、「形の不確かさ最大の場所を触って精度向上」（占有確率の分散が最大の点、`unc`、Act-VH の基準の再実装）とランダム（`rand`）。

## 主な結果（240 配置、1 回触った時点で掴む）

| 触る点の選び方 | 把持成功率 |
|---|---|
| 成功確率の期待値が最大（提案） | 95.4% |
| 形の不確かさ最大の場所を触って精度向上 | 88.8% |
| ランダム | 88.8% |

- 提案 − 形の不確かさ最大の場所を触って精度向上：+6.7 ポイント（95% 区間 [+2.1, +11.3]、McNemar p = 0.005）。差は見えにくいほど大きい（見えている長さ 10mm で +15.0）。
- 触らずに掴むと 72.5%（120 配置）。成功確率が閾値を超えたら触るのをやめる規則は、毎回 1 回触る場合と差がなかった。
- 触った後は天板の外へ腕を戻さず、親指も上げたまま掴む点へ移る（`shelf.DIRECT`）。掴み終わるまでの時間（中央値）は、1回触る場合 17.7 → 15.2 秒、2回触る場合 24.3 → 17.7 秒に短くなった（毎回腕を戻す動きとの比較、20 配置）。

## 成果物（`docs/`）

| ファイル | 内容 |
|---|---|
| `docs/report.pdf` | 提出したレポート（14ページ） |
| `docs/paper_ja.md` | レポートの本文（日本語の Markdown）。図は `docs/figures/` を参照 |
| `docs/code_guide.html` | コードの解説（左にコード、右に関数ごとの動作）。ダウンロードしてブラウザで開く |
| `docs/figures/` | レポートの図1〜5。`fig1_shapes`＝図1（形の族）、`fig0_pipeline`＝図2（学習と推論の流れのブロック図）、`fig2_touch_rule`＝図3（触る点の選び方の比較）、`fig3_layout64`＝図4（配置64の3コマ）、`fig4_calibration`＝図5（予測の較正）。ファイル名の番号はブロック図を足す前のもの |
| `docs/videos/touch_selection.mp4` | 紹介動画（2分27秒） |
| `docs/videos/trial64_voi.mp4`・`trial64_unc.mp4` | 配置64の試行（提案手法／形の不確かさ最大の場所を触って精度向上） |
| `docs/videos/trial64_voi_close.mp4`・`trial64_unc_close.mp4` | 同じ試行の手元のアップ（横から。部品が机の手前に落ちるところまで入る） |
| `docs/videos/trial64_voi_macro.mp4`・`trial64_unc_macro.mp4` | 同じ試行の指先のアップ（どこを触ったかが分かる） |

図と動画の作り直し（リポジトリの直下で）：

```bash
uv run python tools/fig_shapes.py docs/figures/fig1_shapes.png       # 図1（数十秒）
uv run python tools/fig_pipeline.py                                  # 図2 ブロック図（playwright と Chromium が要る：uv pip install playwright; uv run playwright install chromium）
uv run python tools/fig_layout64.py docs/figures/fig3_layout64.png   # 図4（2つの試行を動かすので約5分）
uv run python view_trial.py 64 --rule voi --video docs/videos/trial64_voi.mp4
uv run python view_trial.py 64 --rule unc --video docs/videos/trial64_unc.mp4
uv run python view_trial.py 64 --rule voi --close --video docs/videos/trial64_voi_close.mp4
uv run python view_trial.py 64 --rule unc --close --video docs/videos/trial64_unc_close.mp4
uv run python view_trial.py 64 --rule voi --macro --video docs/videos/trial64_voi_macro.mp4
uv run python view_trial.py 64 --rule unc --macro --video docs/videos/trial64_unc_macro.mp4
uv run python tools/make_video.py                                    # 紹介動画（図1〜3と上の試行の動画から）
```

図3と図5はレポートの文書のグラフ機能で描いたもので、数値は `analyze_touchsel.py`・`analyze_branch.py` の出力です。
`tools/fig_shapes.py` は元の図を後から作り直したもので、視点や寸法線の位置が少し違います（内容は同じ）。

## 環境構築

```bash
uv sync                      # Python 3.11、依存は pyproject.toml
mkdir -p third_party
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie checkout c96a32d28fb5da84da38c1da4d749e7a13212855
```

学習済みの予測器 `predictor.pkl` は scikit-learn 1.8.0 で保存しています。
動画・図の字幕に日本語のフォント（Noto Sans CJK）を使います。Ubuntu・WSL では `sudo apt install fonts-noto-cjk` で入れてください。
画面のない Linux では、カメラの描画に EGL を自動で使います（`build_scene.py` の先頭）。Windows・Mac では設定は不要です。

## ファイル

| ファイル | 役割 |
|---|---|
| `build_scene.py` | G1＋Dex3、机、棚の天板、部品（フランジ＋本体）のシーンを組み立てる |
| `grasp_test.py`・`ik_mink.py` | 指の開閉と挟む点の定義、mink による右腕の逆運動学 |
| `shelf.py` | 棚の下での接近・触る動作・把持・持ち上げと成否判定 |
| `perception.py` | 頭部カメラの観測、粒子の初期化、深度による本体の検出と向きの推定 |
| `partfilter.py` | 触った結果による粒子の更新、成功確率の計算、掴む点と触る点の選択 |
| `gen_grasp_data.py` | 予測器の学習データ（摩擦・質量・指のゲインを振った把持試行） |
| `train_predictor.py` | MLP の学習と較正の確認 → `predictor.pkl` |
| `run_touchsel.py` | 触る点の選び方の比較（voi / unc / rand） |
| `run_branch.py` | 触る回数の比較（0〜4 回触った時点で掴む分岐を状態の複製で評価） |
| `analyze_touchsel.py`・`analyze_branch.py` | 論文の表と数値の集計 |
| `view_trial.py` | 1配置の試行を表示ウィンドウで見る・動画に保存する |
| `run_vlm_compare.py` | 部品2つ並びの場面で、触らずに掴む場合と1回触ってから掴む場合を比べる |
| `run_vlm_point.py` | 掴む位置を SmolVLM に選ばせる場合と、予測器・提案手法で選ぶ場合を比べる |
| `vlm_pick.py` | 指示文で取る部品を SmolVLM に選ばせ、選んだ部品を触って掴むデモ（本実験の数字には関わらない） |
| `tools/` | 図1・図3・紹介動画を作るスクリプト |
| `results/` | `gdata_*`（学習データ 8,000 件）、`tsel_*`（240 配置×3 規則）、`branch_*`（120 配置）。`v1_retract/` は毎回腕を戻す前の動きでの結果 |

## 動きを見る

`view_trial.py` は、1配置の試行（見る → 触る → 掴む）を MuJoCo の表示ウィンドウで再生します。
`run_touchsel.py` と同じ配置・同じ乱数で動かすので、成否は保存済みの結果と一致します（最後に照合して表示）。
触る点（赤）、掴む点（青）、形の候補ごとの本体の手前の縁（黄）を描きます。

```bash
uv run python view_trial.py --list                 # voi が成功し unc が失敗した配置の一覧
uv run python view_trial.py 64 --rule voi          # 表示ウィンドウで見る（Mac は uv run mjpython view_trial.py ...）
uv run python view_trial.py 64 --rule unc
uv run python view_trial.py 64 --rule voi --video trial64_voi.mp4   # 動画に保存（表示ウィンドウは開かない）
```

## 指示文で取る部品を選ぶ（SmolVLM との組み合わせ）

`vlm_pick.py` は、棚の下に色と隠れたサイズの違う部品を2つ並べ（今の位置と右へ 10cm）、指示文（英語）と頭部カメラの画像から
SmolVLM が「どちらを取るか」を番号で答え、選んだ部品を今の方法で触って掴みます。
天板の前端は共通なので、カメラにはどちらもフランジの手前が同じ長さだけ見え、サイズの違い（既定は左がフランジ 12mm・本体 19mm、右が 28mm・13mm、`--sizes` で変更）は見えません。
見える色は SmolVLM が、見えないサイズは触覚で確かめる分担です。
SmolVLM が「何を取るか」、物理から学んだ成功確率が「どこを触ってどこを挟むか」を決める分担です（SayCan と同じ骨組み）。
左隣 5cm に部品があると手が当たって掴めなかったため、間隔は 10cm にしています。

```bash
uv pip install transformers torch pillow           # GPU を使うなら CUDA 版の torch
uv run python vlm_pick.py --instruct "Pick up the green part."                       # 表示ウィンドウで見る
uv run python vlm_pick.py --instruct "Pick up the green part." --video vlm_pick.mp4  # 動画に保存
uv run python vlm_pick.py --answer 2 --video test.mp4    # SmolVLM を使わず答えを手で与えて試す
```

**比べた結果（`run_vlm_compare.py`、`results/summ_vlmcmp.txt`）**：この2つ並びの場面で、選ばれた部品を
触らずに頭部カメラの推定だけで掴む（SmolVLM で選んでそのまま掴む）場合と、提案手法で1回触って確かめてから掴む場合を、
60配置 × 部品2つ＝120回で比べた（部品ごとの隠れたサイズは学習範囲から乱数、同じ観測から両方を評価）。

| 掴み方 | 成功率 |
|---|---|
| 触らずに掴む | 75.8% |
| 1回触って確かめてから掴む | 95.0% |

差は +19.2 ポイント（95% 信頼区間 [+11.7, +26.7]、片方だけ成功 25 対 2、McNemar の正確検定 p < 0.001）。
見えにくいほど差が大きい（見え方 10mm で 63% 対 93%）。SmolVLM が正しい部品を選べるかはこの比較に含めておらず、
`vlm_pick.py` で別に確かめる。

```bash
python3 run_vlm_compare.py 0 2 60 & python3 run_vlm_compare.py 1 2 60 & wait   # 2並列で約6分
python3 run_vlm_compare.py --summary
```

**掴む位置まで SmolVLM に任せる場合（`run_vlm_point.py`）**：同じ120試行で、カメラの推定の平均の部品軸に沿って候補点 A〜F
（手前の面から 8〜48mm）を画像に描き、SmolVLM に掴む点を選ばせる（学習なし）。腕の動かし方は同じ逆運動学。
候補点ごとの成否は先にシミュレーションで全部求めてあるので（`results/vlmpoint_*.jsonl`、画像は `results/vlmpoint_img/`）、
SmolVLM を動かす PC では MuJoCo の計算は要らない。

```bash
python3 run_vlm_point.py sim 0 2 60 & python3 run_vlm_point.py sim 1 2 60 & wait   # 候補点ごとの成否と画像（2並列で約40分、済み）
python3 run_vlm_point.py ask           # SmolVLM に答えさせる → results/vlmpoint_answers.jsonl
python3 run_vlm_point.py summary       # A（SmolVLM）・B（触らずに予測器）・C（触ってから）を比べる
```

いつも同じ候補点を掴んだときの成功率は A 0%・B 18%・C 57%・D 65%・E 32%・F 1%（候補点のどれかで掴める割合 98.3%）。

結果（`results/summ_vlmpoint.txt`、答えは `results/vlmpoint_answers*.jsonl`、RTX 3060 Ti で実行）：

| 掴む位置の決め方 | 成功率（120回） | 答えの内訳 |
|---|---|---|
| SmolVLM-500M、知識なしの問い | 0.8% | A 53・F 67 |
| SmolVLM-2.2B（SmolVLM-Instruct）、知識なし | 0.8% | F 120 |
| SmolVLM-500M、知識ありの問い（`--prompt informed`） | 43.3% | C 80・E 23・ほか 17 |
| SmolVLM-2.2B、知識あり | 61.7% | C 69・D 51 |
| 物理から学んだ予測器（触らない） | 75.8% | — |
| 提案手法（1回触ってから） | 95.0% | — |

知識なしの問いでは両端の点ばかり答え、画像を使った判断になっていない。部品の長さの範囲・候補点の距離・挟む余裕を文で与えると
中ほどの点を選ぶようになるが、2.2B でも答えは C と D だけで、いつも C（57%）・D（65%）を掴むのと同程度。
予測器（触らない）は 2.2B・知識ありより +14.2 ポイント（片方だけ成功 32 対 15、McNemar p = 0.019）、
提案手法は +33.3 ポイント（44 対 4、p < 0.001）高い。

比較動画 `docs/videos/vlm_compare.mp4`（約49秒）は、配置0の部品2（触らずに掴むと失敗、1回触ると成功した例）を `vlm_pick.py --touches 0／1` で撮り、`tools/make_vlm_video.py` で左右に並べたもの。SmolVLM の答えは手で与えている（動画内に明記）。

既定のモデルは `HuggingFaceTB/SmolVLM-500M-Instruct`（`--model` で変更、GPU があれば自動で使う）。
SmolVLM に渡した画像は `vlm_input.png`、問いと答えは `vlm_answer.json` に保存します。

## 再現

結果ファイルからの集計（数秒）：

```bash
python analyze_touchsel.py     # 触る点の選び方の比較
python analyze_branch.py       # 触る回数の比較、較正、失敗時の部品の移動
python train_predictor.py      # 予測器の学習（results/gdata_*.jsonl から）
```

実験をやり直す場合（引数は 担当番号 k・並列数 n・配置数 N。k = 0..n−1 を並列に走らせる）：

```bash
python gen_grasp_data.py 0 2 8000
python run_touchsel.py 0 2 240 predictor.pkl
python run_branch.py 0 2 120 predictor.pkl
```

配置は乱数の種で固定しているので、同じ環境なら同じ結果になります（例：`run_touchsel` の配置 0 を再実行し、保存済みの結果と一致することを確認済み）。
結果は `results/` に追記され、途中で止めても続きから再開します。置き場所は環境変数 `RESULTS_DIR` で変えられます。

前の動き（触るたびに天板の外へ腕を戻し、親指を掴む形に戻す）の結果は `results/v1_retract/` にあります。`shelf.py` の `DIRECT = False` にすると、その動きで同じ結果が出ます（集計は `RESULTS_DIR=results/v1_retract python analyze_touchsel.py`）。

## 参考文献

番号はレポートの参考文献に合わせています。

- [1] L. Rustler et al., "Active Visuo-Haptic Object Shape Completion," RA-L 2022. https://arxiv.org/abs/2203.09149
- [3] L. Rustler, M. Hoffmann, "ShapeGrasp: Simultaneous Visuo-Haptic Shape Completion and Grasping for Improved Robot Manipulation," 2026. https://arxiv.org/abs/2605.02347
- [5] J. Lundell, F. Verdoja, V. Kyrki, "Robust Grasp Planning Over Uncertain Shape Completions," IROS 2019. https://arxiv.org/abs/1903.00645
- [7] Wang et al., "DA-GRD: Decision-Aware Grasp-Relevant Disambiguation for Tactile Recovery under Perception-to-Execution Mismatches," 2026.
- [12] J. Mahler et al., "Dex-Net 2.0," RSS 2017. https://arxiv.org/abs/1703.09312
- [17] E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A Physics Engine for Model-Based Control," IROS 2012.
- [18] MuJoCo Menagerie. https://github.com/google-deepmind/mujoco_menagerie
- [19] mink. https://github.com/kevinzakka/mink

## 今後の作業（arXiv 投稿前）

- 参考文献 [7] DA-GRD、[9] TACTFUL、[14] Jiang ら、[15] Caddeo ら、[16] DexTacWAM の著者名と arXiv 番号、[10] TactoFind の著者名を原文で補う。
- 原稿を英語化し、IEEE 会議形式の LaTeX にする。
