"""SmolVLM に掴む位置まで任せる比較（条件 A）。run_vlm_compare.py と同じ場面・同じカメラの推定で、
  A：SmolVLM が、画像に描いた候補点（A〜F）から掴む位置を選ぶ（学習なし）
  B：物理から学んだ予測器で、触らずにカメラの推定だけから掴む位置を選ぶ（run_vlm_compare の notouch）
  C：提案手法で1回触ってから掴む位置を選ぶ（run_vlm_compare の touch）
を比べる。腕の動かし方（逆運動学）はすべて同じで、違うのは掴む位置の決め方だけ。

候補点は、カメラの推定の平均の部品軸に沿って、手前の面から 8・16・24・32・40・48mm。
各候補点で状態を複製して掴み（run_branch.branch_grasp）、成否を全部記録しておくので、SmolVLM の答えは後から当てはめるだけでよい
（SmolVLM を動かす PC に MuJoCo の計算は要らない）。

使い方:
  1. シミュレーション（候補点ごとの成否と、SmolVLM に渡す画像）:
       python3 run_vlm_point.py sim k n N        → results/vlmpoint_k.jsonl、results/vlmpoint_img/<配置>_<部品>.png
  2. SmolVLM に答えさせる（GPU があれば使う）:
       python3 run_vlm_point.py ask [--model HuggingFaceTB/SmolVLM-500M-Instruct]   → results/vlmpoint_answers.jsonl
  3. 集計:
       python3 run_vlm_point.py summary
"""
import os, json, glob, gc, argparse
import numpy as np

DEPTHS_MM = (8, 16, 24, 32, 40, 48)
LETTERS = "ABCDEF"
IMG_DIR = "results/vlmpoint_img"
QUESTION = ("A robot hand will pinch this part from above and lift it. The front of the part is visible, but the rest is "
            "hidden under the shelf board. Grasping only the thin flat front lip will not lift the part; the hand must "
            "pinch the taller main body. The letters A to F mark candidate grasp points along the part, from front (A) to "
            "back (F). Which point should the hand grasp? Answer with one letter only.")


def candidates(est, F):
    xf, y = est[:, 0].mean(), est[:, 1].mean()
    yaw = np.arctan2(np.sin(est[:, 2]).mean(), np.cos(est[:, 2]).mean())
    return [(xf + np.cos(yaw) * t / 1000, y + np.sin(yaw) * t / 1000) for t in DEPTHS_MM]


def draw_image(s, pts, z, path):
    """頭部カメラの画像から部品のまわりを切り出して4倍にし、候補点に A〜F を描く。"""
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import perception as P, view_trial as V
    cam = P.HeadCam(s.m, s.d)
    r = mujoco.Renderer(s.m, 480, 640); r.update_scene(s.d, "head_cam"); rgb = r.render(); r.close()
    uv = cam.project([[x, y, z] for x, y in pts])
    cu, cv = uv.mean(0)
    u0, v0 = int(np.clip(cu - 45, 0, 640 - 90)), int(np.clip(cv - 40, 0, 480 - 80))   # 画像の端でも 90×80 を切り出す
    sc = 4
    im = Image.fromarray(rgb[v0:v0 + 80, u0:u0 + 90]).resize((90 * sc, 80 * sc), Image.LANCZOS)
    d = ImageDraw.Draw(im); font = ImageFont.truetype(V.FONT, 18, index=0)
    for k, (u, v) in enumerate(uv):
        x, y = (u - u0) * sc, (v - v0) * sc
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 0), outline=(0, 0, 0))
        tx = x + (14 if k % 2 == 0 else -28)
        d.rectangle([tx - 2, y - 12, tx + 14, y + 10], fill=(255, 255, 255))
        d.text((tx, y - 13), LETTERS[k], font=font, fill=(0, 0, 0))
    im.save(path)


def trial(i, slot):
    import shelf as S, perception as P, partfilter as F, build_scene as bs, run_branch as RB
    import vlm_pick as VP, run_vlm_compare as C
    sizes = C.sizes_for(i)
    s, lay = VP.make_sim(i, slot, C.COLORS, sizes)
    rng = np.random.default_rng(5000 + i)                     # run_vlm_compare と同じ乱数 → 同じカメラの推定
    F.use_predictor("predictor.pkl", s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    est, _ = P.init_particles_part(cam, cm, occl, RB.W, ztab, rng, (S.X_FRONT, S.Y0 + VP.SLOTS[slot]),
                                   RB.FL_R, RB.L_R, yaw0=yaw0)
    est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, RB.FL_R, rng)
    pts = candidates(est, F)
    os.makedirs(IMG_DIR, exist_ok=True)
    draw_image(s, pts, ztab + bs.FLANGE_H, f"{IMG_DIR}/{i}_{slot}.png")   # 点はフランジの上面の高さに描く
    succ = [bool(RB.branch_grasp(s, [x, y, ztab + RB.W / 2])["success"]) for x, y in pts]
    fl, L = sizes[slot]
    out = dict(i=i, slot=slot, vis=lay["vis"] * 1000, body_mm=[fl * 1000, (fl + L) * 1000], body_seen=seen,
               depths_mm=list(DEPTHS_MM), success=succ)
    del s; gc.collect()
    return out


def ask(model_name):
    from PIL import Image
    fn = "results/vlmpoint_answers.jsonl"
    done = {json.loads(l)["img"] for l in open(fn)} if os.path.exists(fn) else set()
    out = open(fn, "a")
    model = None
    imgs = sorted(glob.glob(f"{IMG_DIR}/*.png"))
    print(f"画像 {len(imgs)} 枚、答え済み {len(done)} 枚", flush=True)
    if not imgs:
        raise SystemExit(f"{IMG_DIR} に画像がありません（python3 run_vlm_point.py sim で作る）")
    for p in imgs:
        name = os.path.basename(p)
        if name in done:
            continue
        q, ans, dev, model = ask_one(Image.open(p).convert("RGB"), model_name, model)
        m = [c for c in ans.upper() if c in LETTERS]
        out.write(json.dumps(dict(img=name, answer=ans, letter=m[0] if m else None, model=model_name, device=dev)) + "\n")
        out.flush(); print(name, repr(ans), flush=True)


def ask_one(image, model_name, cache):
    import torch, transformers
    from transformers import AutoProcessor
    from packaging.version import Version
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if cache is None:
        dtype = torch.bfloat16 if dev == "cuda" else torch.float32
        kw = {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}
        cache = (AutoProcessor.from_pretrained(model_name), AutoVLM.from_pretrained(model_name, **kw).to(dev).eval())
    proc, model = cache
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[image], return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    ans = proc.batch_decode(o[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return QUESTION, ans, dev, cache


def summary():
    R = {(r["i"], r["slot"]): r for f in glob.glob("results/vlmpoint_*.jsonl") if "answers" not in f
         for r in map(json.loads, open(f))}
    Cm = {(r["i"], r["slot"]): r for f in glob.glob("results/vlmcmp_*.jsonl") for r in map(json.loads, open(f))}
    print(f"試行 {len(R)}")
    S = np.array([R[k]["success"] for k in sorted(R)], float)
    print("候補点ごとの成功率（いつもその点を掴んだ場合）：" +
          "  ".join(f"{LETTERS[j]}({DEPTHS_MM[j]}mm) {S[:, j].mean() * 100:.0f}%" for j in range(len(DEPTHS_MM))))
    print(f"候補点のどれかで掴める割合（候補点の上限）: {S.max(1).mean() * 100:.1f}%")
    keys = [k for k in sorted(R) if k in Cm]
    if keys:
        print(f"B 触らずに予測器で選ぶ: {np.mean([Cm[k]['notouch']['success'] for k in keys]) * 100:.1f}%   "
              f"C 1回触ってから選ぶ: {np.mean([Cm[k]['touch']['success'] for k in keys]) * 100:.1f}%  （同じ {len(keys)} 試行）")
    fn = "results/vlmpoint_answers.jsonl"
    if not os.path.exists(fn):
        print("SmolVLM の答えはまだありません（python3 run_vlm_point.py ask）"); return
    A = {json.loads(l)["img"]: json.loads(l) for l in open(fn)}
    rows = []
    for k in sorted(R):
        a = A.get(f"{k[0]}_{k[1]}.png")
        if a is None:
            continue
        j = LETTERS.find(a["letter"]) if a["letter"] else -1
        rows.append((k, j, R[k]["success"][j] if j >= 0 else False))
    if not rows:
        print("答えと結果の組がありません"); return
    sa = np.array([r[2] for r in rows], float)
    from collections import Counter
    cnt = Counter(LETTERS[r[1]] if r[1] >= 0 else "読めない" for r in rows)
    print(f"A SmolVLM が掴む位置を選ぶ: {sa.mean() * 100:.1f}%（{len(rows)} 試行）  答えの内訳: {dict(sorted(cnt.items()))}")
    ks = [r[0] for r in rows if r[0] in Cm]
    if ks:
        a_ = np.array([r[2] for r in rows if r[0] in Cm], float)
        for name, key in (("B", "notouch"), ("C", "touch")):
            b_ = np.array([Cm[k][key]["success"] for k in ks], float)
            nb, na = int(((b_ == 1) & (a_ == 0)).sum()), int(((a_ == 1) & (b_ == 0)).sum())
            from scipy.stats import binomtest
            p = binomtest(nb, nb + na, 0.5).pvalue if nb + na else 1.0
            print(f"  {name} − A: {(b_ - a_).mean() * 100:+.1f} ポイント（片方だけ成功 {name} {nb}・A {na}、McNemar p = {p:.3g}）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("sim", "ask", "summary"))
    ap.add_argument("args", nargs="*")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    a = ap.parse_args()
    if a.mode == "summary":
        summary()
    elif a.mode == "ask":
        ask(a.model)
    else:
        k, n, N = map(int, a.args)
        os.makedirs("results", exist_ok=True)
        fn = f"results/vlmpoint_{k}.jsonl"
        done = {(json.loads(l)["i"], json.loads(l)["slot"]) for l in open(fn)} if os.path.exists(fn) else set()
        out = open(fn, "a")
        jobs = [(i, sl) for i in range(N) for sl in (0, 1)]
        for j, (i, sl) in enumerate(jobs):
            if j % n != k or (i, sl) in done:
                continue
            out.write(json.dumps(trial(i, sl)) + "\n"); out.flush()
