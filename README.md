# 棚の奥で一部が隠れた部品を掴むための触る点の選択

Physical AI 講座 最終課題のコード・学習済みモデル・実験結果です。
MuJoCo 上の Unitree G1＋Dex3（3指ハンド）が、上の段の天板で奥が隠れた小部品を、
見えた部分と軽く触った結果から形を推定して掴みます。

## 概要

- 部品は手前に低いフランジ（高さ4mm）、奥に掴める本体（幅・高さ20mm）をもつ。フランジの長さ ℓ_f（10〜30mm）と本体の長さ ℓ_b（12〜20mm）は未知。
- 頭部カメラの領域分割と深度から、形の候補 θ = (位置, 向き, ℓ_f, ℓ_b) を 200 個の粒子で推定する。
- 把持成功確率は、候補ごとに小さな MLP（物理シミュレーションの 8,000 回の把持から学習）で出し、候補で平均する。
- **触る点は「触った後に得られる最良の成功確率の期待値」が最大の点**として選ぶ（`voi`）。
  比較として、形の不確かさ（占有確率の分散）が最大の点（`unc`、Act-VH の基準の再実装）とランダム（`rand`）。

## 主な結果（240 配置、1 回触った時点で掴む）

| 触る点の選び方 | 把持成功率 |
|---|---|
| 成功確率の期待値が最大（提案） | 95.8% |
| 形の不確かさが最大 | 90.0% |
| ランダム | 88.3% |

- 提案 − 不確かさ最大：+5.8 ポイント（95% 区間 [+1.7, +9.6]、McNemar p = 0.007）。差は見えにくいほど大きい（見えている長さ 10mm で +13.3）。
- 触らずに掴むと 72.5%（120 配置）。成功確率が閾値を超えたら触るのをやめる規則は、毎回 1 回触る場合と差がなかった。

## 環境構築

```bash
uv sync                      # Python 3.11、依存は pyproject.toml
mkdir -p third_party
git clone https://github.com/google-deepmind/mujoco_menagerie third_party/mujoco_menagerie
git -C third_party/mujoco_menagerie checkout c96a32d28fb5da84da38c1da4d749e7a13212855
```

学習済みの予測器 `predictor.pkl` は scikit-learn 1.8.0 で保存しています。
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
| `results/` | `gdata_*`（学習データ 8,000 件）、`tsel_*`（240 配置×3 規則）、`branch_*`（120 配置） |

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
結果は `results/` に追記され、途中で止めても続きから再開します。

## 参考文献

番号は論文原稿の参考文献に合わせています。

- [1] L. Rustler et al., "Active Visuo-Haptic Object Shape Completion," RA-L 2022. https://arxiv.org/abs/2203.09149
- [3] L. Rustler, M. Hoffmann, "ShapeGrasp: Simultaneous Visuo-Haptic Shape Completion and Grasping for Improved Robot Manipulation," 2026. https://arxiv.org/abs/2605.02347
- [5] J. Lundell, F. Verdoja, V. Kyrki, "Robust Grasp Planning Over Uncertain Shape Completions," IROS 2019. https://arxiv.org/abs/1903.00645
- [7] Wang et al., "DA-GRD: Decision-Aware Grasp-Relevant Disambiguation for Tactile Recovery under Perception-to-Execution Mismatches," 2026.
- [12] J. Mahler et al., "Dex-Net 2.0," RSS 2017. https://arxiv.org/abs/1703.09312
- [17] E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A Physics Engine for Model-Based Control," IROS 2012.
- [18] MuJoCo Menagerie. https://github.com/google-deepmind/mujoco_menagerie
- [19] mink. https://github.com/kevinzakka/mink
