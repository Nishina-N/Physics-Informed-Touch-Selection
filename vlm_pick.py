"""指示文で取る部品を選び（SmolVLM）、選んだ部品を今の方法で触って掴む（組み合わせのデモ）。

棚の下に色の違う部品を2つ並べる（今の位置と、そこから右へ 10cm）。
  1. 頭部カメラの画像から部品のまわりを切り出して拡大し、部品の上に番号 1・2 を書く
  2. SmolVLM に画像と指示文（例："Pick up the green part."）を渡し、どの番号かを答えさせる
  3. 答えた部品について、view_trial.py と同じ「見る → 触る → 掴む」を行う（触る点は voi）
役割の分担は SayCan と同じ：言葉と画像から「何を取るか」を SmolVLM が決め、「どこを触ってどこを挟むか」は
物理から学んだ成功確率で決める。本実験（run_touchsel.py）の数字には関わらない、デモのためのコード。

使い方（リポジトリの直下で）:
  SmolVLM を入れる  : uv pip install transformers torch pillow   （GPU を使うなら CUDA 版の torch）
  見る              : uv run python vlm_pick.py --instruct "Pick up the green part."
  動画に保存        : uv run python vlm_pick.py --instruct "Pick up the green part." --video docs/videos/vlm_pick.mp4
  SmolVLM なしで試す: uv run python vlm_pick.py --answer 2 --video /tmp/test.mp4   （答えの番号を手で与える）
  色の並べ方        : --colors red,blue（左から。既定は blue,green）
  モデル            : --model HuggingFaceTB/SmolVLM-500M-Instruct（既定）。GPU があれば自動で使う
SmolVLM に渡した画像は vlm_input.png、問いと答えは vlm_answer.json に保存する（--out で置き場所を変える）。
"""
import argparse, json, os, re
import numpy as np
import shelf as S
import mujoco
import perception as P
import partfilter as F
import build_scene as bs
import run_branch as RB
import run_touchsel as T
import view_trial as V

SLOTS = (0.0, -0.10)             # 部品を置く左右の位置（S.Y0 からのずれ[m]、負が右）。
# 左へずらすと掴めず、5cm 間隔では左隣の部品に手が当たって掴めなかったので、10cm 離した2つにした
COLORS = {"red": (0.85, 0.15, 0.15), "green": (0.15, 0.65, 0.25), "blue": (0.15, 0.35, 0.90),
          "yellow": (0.95, 0.80, 0.10), "white": (0.92, 0.92, 0.92), "black": (0.10, 0.10, 0.10)}
COLOR_JA = {"red": "赤", "green": "緑", "blue": "青", "yellow": "黄", "white": "白", "black": "黒"}
QUESTION = ("The image shows two parts under a shelf, labeled with the numbers 1 and 2. "
            "Instruction: {instruct} Which numbered part should the robot pick up? Answer with the number only.")


def add_extras(spec, lay, target_slot, colors):
    """target_slot 以外の位置に、同じ形の部品を固定で置く（腕が当たっても動かない置物。触る・掴むのは target だけ）。"""
    W, fl, L = RB.W, lay["fl"], lay["L"]
    h = np.radians(lay["yaw"]) / 2
    for k, dy in enumerate(SLOTS):
        if k == target_slot:
            continue
        cx = S.X_FRONT + fl + L / 2
        b = spec.worldbody.add_body(name=f"extra_{k}", pos=[cx, S.Y0 + lay["dy"] + dy, bs.TABLE_TOP_Z + W / 2 + 0.001],
                                    quat=[np.cos(h), 0, 0, np.sin(h)])
        rgba = [*COLORS[colors[k]], 1]
        b.add_geom(name=f"extra_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, W / 2, W / 2], rgba=rgba)
        f = fl + 0.001
        b.add_geom(name=f"extra_{k}_flange", type=mujoco.mjtGeom.mjGEOM_BOX, size=[f / 2, W / 2, bs.FLANGE_H / 2],
                   pos=[-L / 2 - fl + f / 2, 0, -W / 2 + bs.FLANGE_H / 2], rgba=rgba)
    return spec


def make_sim(i, target_slot, colors):
    """配置 i の部品を target_slot に置き、ほかの位置に色違いの置物を置いた ShelfSim。"""
    lay = RB.layout(i)
    orig = bs.build_shelf
    bs.build_shelf = lambda *a, **k: add_extras(orig(*a, **k), lay, target_slot, colors)
    try:
        s = S.ShelfSim(RB.W, visible=lay["vis"], part=(lay["fl"], lay["L"]),
                       dxy=(0, lay["dy"] + SLOTS[target_slot]), yaw_deg=lay["yaw"])
    finally:
        bs.build_shelf = orig
    m = s.m
    for gname in (s.cube, s.cube + "_flange"):
        m.geom_rgba[m.geom(gname).id] = [*COLORS[colors[target_slot]], 1]
    RB.apply_phys(s, lay)
    return s, lay


def vlm_image(s, lay):
    """頭部カメラの画像から部品のまわりを切り出して3倍に拡大し、部品の上に番号を書く。"""
    from PIL import Image, ImageDraw, ImageFont
    cam = P.HeadCam(s.m, s.d)
    r = mujoco.Renderer(s.m, 480, 640); r.update_scene(s.d, "head_cam"); rgb = r.render(); r.close()
    z = bs.TABLE_TOP_Z + RB.W
    pts = [[S.X_FRONT + 0.005, S.Y0 + lay["dy"] + dy, z] for dy in SLOTS]
    uv = cam.project(pts)
    u0, u1 = int(min(uv[:, 0]) - 60), int(max(uv[:, 0]) + 60)
    v0, v1 = int(min(uv[:, 1]) - 70), int(max(uv[:, 1]) + 40)
    u0, v0 = max(u0, 0), max(v0, 0)
    sc = 3
    im = Image.fromarray(rgb[v0:v1, u0:u1]).resize(((u1 - u0) * sc, (v1 - v0) * sc), Image.LANCZOS)
    d = ImageDraw.Draw(im); font = ImageFont.truetype(V.FONT, 40, index=0)
    for k, (u, v) in enumerate(uv):
        x, y = (u - u0) * sc, (v - v0) * sc - 80
        d.rectangle([x - 22, y - 4, x + 22, y + 48], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        d.text((x - 11, y - 2), str(k + 1), font=font, fill=(0, 0, 0))
    return im


def ask_smolvlm(image, instruct, model_name):
    """SmolVLM に画像と指示文を渡し、答え（文字列）を返す。"""
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:                                       # 古い transformers
        from transformers import AutoModelForVision2Seq as AutoVLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(model_name)
    import transformers
    from packaging.version import Version
    kw = {"dtype": dtype} if Version(transformers.__version__) >= Version("4.56") else {"torch_dtype": dtype}
    model = AutoVLM.from_pretrained(model_name, **kw).to(dev).eval()
    q = QUESTION.format(instruct=instruct)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[image], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    return q, ans, dev


def parse_answer(text):
    m = re.search(r"[12]", text)
    return None if m is None else int(m.group()) - 1


def pick(i, slot, colors, instruct, answer_text, vlm_img, video=None, out=".", n_touch=1):
    """view_trial.run と同じ手順で、選んだ部品を触って掴む。事前分布の中心だけ選んだ位置へずらす。"""
    s, lay = make_sim(i, slot, colors)
    rng = np.random.default_rng(5000 + i); rr = np.random.default_rng(9000 + i)
    F.use_predictor("predictor.pkl", s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    prior = (S.X_FRONT, S.Y0 + SLOTS[slot])
    est, _ = P.init_particles_part(cam, cm, occl, RB.W, ztab, rng, prior, RB.FL_R, RB.L_R, yaw0=yaw0)
    est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, RB.FL_R, rng)
    u, v = cam.project([[S.X_FRONT, S.Y0 + lay["dy"] + sum(SLOTS) / len(SLOTS), s.c0[2]]])[0]   # 小窓：部品が全部入るよう間を中心に
    head_crop = (int(np.clip(u - 107, 0, 640 - 214)), int(np.clip(v - 95, 0, 480 - 160)))
    m.geom_rgba[m.geom("board").id] = [0.78, 0.80, 0.86, 0.75]
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    m.light_castshadow[:] = 0
    head = f'instruction: "{instruct}"   SmolVLM: "{answer_text}" -> part {slot + 1} ({colors[slot]})'
    viewer = None
    if video is None:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    sh = V.Show(s, viewer, video, 1.0, head_crop)
    if viewer is not None:
        with viewer.lock():
            sh._aim(viewer.cam)
    if video and sh.frames is not None:
        sh.frames += [intro_frame(vlm_img, instruct, answer_text, slot, colors)] * (V.FPS * 6)
    sh.update(est=est, text=f"{head}\nlook: body seen by camera: {'yes' if seen else 'no'}")
    for k in range(1, n_touch + 1):
        gc_ = F.axis_points(est); tc = F.axis_points(est, step=0.003)
        tx, ty = T.choose("voi", est, gc_, tc, rr)
        sh.update(touch=(tx, ty), text=f"{head}\ntouch {k}/{n_touch}: moving to the red point")
        r = s.touch([tx, ty])
        est = F.update(est, tx, ty, r["obs"], rng)
        sh.update(est=est, text=f"{head}\ntouch {k}/{n_touch}: {V.OBS_NAME[r['obs']]}")
    (gx, gy), p = F.best_grasp_m(est, F.axis_points(est))
    sh.update(grasp=(gx, gy), text=f"{head}\ngrasp at the blue point   predicted success {p:.2f}")
    g = s.grasp([gx, gy, ztab + RB.W / 2])
    sh.update(text=f"{head}\n{'SUCCESS' if g['success'] else 'FAILURE'}   lifted {g['rise_cm']:.1f} cm")
    s.sim.step(1.0)
    if video and sh.frames is not None:
        import imageio.v2 as imageio
        imageio.mimsave(video, sh.frames, fps=V.FPS); sh.rd.close(); sh.rc.close(); sh.hc.close()
        print(f"保存: {video}（{len(sh.frames)} コマ）")
    if viewer is not None:
        print("表示ウィンドウを閉じると終了します")
        import time
        while viewer.is_running():
            time.sleep(0.1)
    return g["success"]


def intro_frame(vlm_img, instruct, answer_text, slot, colors):
    """動画の最初：SmolVLM に渡した画像と、指示文・答え（1072×640）。"""
    from PIL import Image, ImageDraw, ImageFont
    font = lambda n: ImageFont.truetype(V.FONT, n, index=0)
    im = Image.new("RGB", (V.VW, V.VH), (0, 0, 0)); dr = ImageDraw.Draw(im)
    p = vlm_img.copy(); p.thumbnail((V.VW - 32, V.VH - 200))
    im.paste(p, ((V.VW - p.width) // 2, 150))
    dr.text((16, 12), "SmolVLM が指示文と頭部カメラの画像から、取る部品を選ぶ", font=font(20), fill=(210, 210, 210))
    dr.text((16, 50), f'指示文："{instruct}"', font=font(24), fill=(255, 255, 255))
    dr.text((16, 92), f'SmolVLM の答え："{answer_text}"  →  部品 {slot + 1}（{COLOR_JA.get(colors[slot], colors[slot])}）',
            font=font(24), fill=(255, 230, 120))
    return np.array(im)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="指示文で取る部品を SmolVLM に選ばせ、選んだ部品を触って掴む")
    ap.add_argument("--layout", type=int, default=0, help="部品の形と見え方（run_branch.layout の番号）")
    ap.add_argument("--colors", help="左から2つの色（例 red,blue）。既定は blue,green")
    ap.add_argument("--instruct", default="Pick up the green part.", help="指示文（英語）")
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--answer", type=int, choices=(1, 2), help="SmolVLM を使わず、答えの番号を手で与える")
    ap.add_argument("--video", help="動画のファイル名（与えると表示ウィンドウを開かずに保存する）")
    ap.add_argument("--out", default=".", help="SmolVLM に渡した画像と答えの置き場所")
    a = ap.parse_args()
    if a.colors:
        colors = a.colors.split(",")
    else:
        colors = ["blue", "green"]
    s0, lay = make_sim(a.layout, 0, colors)                  # 選ぶ前の場面（どの位置でも見た目は同じ）
    img = vlm_image(s0, lay); del s0
    os.makedirs(a.out, exist_ok=True); img.save(os.path.join(a.out, "vlm_input.png"))
    if a.answer:
        q, ans, dev = QUESTION.format(instruct=a.instruct), str(a.answer), "manual"
    else:
        q, ans, dev = ask_smolvlm(img, a.instruct, a.model)
    slot = parse_answer(ans)
    rec = dict(layout=a.layout, colors=colors, instruct=a.instruct, question=q, answer=ans, device=dev,
               model=None if a.answer else a.model, slot=slot)
    json.dump(rec, open(os.path.join(a.out, "vlm_answer.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rec, ensure_ascii=False))
    if slot is None:
        raise SystemExit(f"SmolVLM の答え「{ans}」から番号を読み取れませんでした")
    pick(a.layout, slot, colors, a.instruct, ans, img, a.video, a.out)
