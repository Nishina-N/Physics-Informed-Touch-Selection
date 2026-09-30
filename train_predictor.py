"""形の候補ごとの成功確率を予測する小さなMLPを学習し、較正を確かめる。
入力：u（本体の手前の縁から挟む点まで）、v（挟む点から本体の奥の縁まで）、b（横のずれ）、yaw（部品の向き）、
      L（本体の長さ）、reach（天板の前端から挟む点までの奥行き）。単位は mm と度。
出力：成功確率。物理のばらつき（摩擦・質量・指のゲイン）は学習データの中で平均される。
使い方: python3 train_predictor.py → predictor.pkl と較正の表
"""
import json, glob, pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEAT = ["u", "v", "b", "yaw", "L", "reach"]


def features(r):
    return [r["u"] * 1000, r["v"] * 1000, r["b"] * 1000, r["yaw"], r["L"] * 1000, r["reach"] * 1000]


def load():
    R = [json.loads(l) for f in sorted(glob.glob("results/gdata_*.jsonl")) for l in open(f)]
    R.sort(key=lambda r: r["i"])
    X = np.array([features(r) for r in R]); y = np.array([r["success"] for r in R], dtype=int)
    return R, X, y


def reliability(p, y, bins=(0, .1, .3, .5, .7, .9, 1.01)):
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (p >= lo) & (p < hi)
        if k.sum():
            rows.append((lo, hi, int(k.sum()), float(p[k].mean()), float(y[k].mean())))
    return rows


if __name__ == "__main__":
    R, X, y = load()
    n = len(y); n_te = n // 5
    Xtr, ytr, Xte, yte = X[:-n_te], y[:-n_te], X[-n_te:], y[-n_te:]     # 後ろの2割を較正の確認に使う
    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3,
                                                        max_iter=2000, early_stopping=True, random_state=0))
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    brier = float(np.mean((p - yte) ** 2)); acc = float(np.mean((p > 0.5) == yte))
    print(f"n={n} 学習{len(ytr)} 確認{len(yte)} 成功率{y.mean():.2f} Brier {brier:.3f} 正解率 {acc:.3f}")
    for lo, hi, k, pm, ym in reliability(p, yte):
        print(f"  予測 {lo:.1f}〜{hi:.1f}: {k}件 予測平均 {pm:.2f} 実際 {ym:.2f}")
    clf.fit(X, y)                                                         # 最終版は全件で学習
    pickle.dump(dict(model=clf, feat=FEAT), open("predictor.pkl", "wb"))
