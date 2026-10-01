"""1配置の試行を目で見る：MuJoCo の表示ウィンドウで再生する、または動画に保存する。

run_touchsel.py と同じ配置・同じ乱数で「見る → 触る（n回）→ 掴む」を行うので、結果は
results/tsel_*.jsonl の同じ配置・同じ規則・同じ回数の成否と一致する（最後に照合して表示する）。
表示用に、触る点（赤）、掴む点（青）、粒子ごとの本体の手前の縁（黄の小さな点）を描く。

使い方（リポジトリの直下で）:
  見る        : uv run python view_trial.py 64 --rule voi
                Mac は表示ウィンドウに mjpython が要る : uv run mjpython view_trial.py 64 --rule voi
  動画に保存  : uv run python view_trial.py 64 --rule voi --video trial64_voi.mp4
  配置を探す  : uv run python view_trial.py --list
                （1回触った時点で voi が成功し unc が失敗した配置の一覧）
"""
import argparse, glob, json, time
import numpy as np
import shelf as S                 # build_scene 経由で描画の設定を先に決める
import mujoco
import perception as P
import partfilter as F
import build_scene as bs
import run_branch as RB           # 配置と物理の値（perception.TEMP も 0.05 になる）
import run_touchsel as T          # 触る点の選び方

OBS_NAME = {0: "miss", 1: "body (fingertip center)", 2: "body (finger pad)", 3: "flange"}
FPS = 30


def saved_results():
    by = {}
    for f in glob.glob("results/tsel_*.jsonl"):
        for l in open(f):
            r = json.loads(l); by.setdefault(r["i"], {})[r["rule"]] = r
    return by


def list_layouts():
    """1回触った時点で voi が成功し unc が失敗した配置（動画の素材の候補）。"""
    by = saved_results()
    print("配置  見え方  voi  unc  rand   （1回触った時点で掴んだ成否）")
    for i in sorted(by):
        v = by[i]
        if len(v) < 3:
            continue
        s = {k: v[k]["steps"][0]["success"] for k in ("voi", "unc", "rand")}
        if s["voi"] and not s["unc"]:
            mark = lambda x: " o " if x else " x "
            print(f"{i:4d}  {v['voi']['vis']:4.0f}mm  {mark(s['voi'])}  {mark(s['unc'])}  {mark(s['rand'])}")


class Show:
    """シミュレーションの1刻みごとに呼ばれ、表示ウィンドウを更新する（または動画のコマを撮る）。"""

    def __init__(self, s, viewer=None, video=None, speed=1.0, head_crop=None):
        self.s, self.viewer, self.speed, self.head_crop = s, viewer, speed, head_crop
        self.n, self.t0 = 0, time.time()
        self.touch_pt, self.grasp_pt, self.fronts, self.text = None, None, None, ""
        self.frames = None
        if video:
            self.frames = []
            s.m.vis.global_.offwidth, s.m.vis.global_.offheight = 960, 544   # 画面外の描画の大きさ（既定は640×480）。高さは動画の符号化に合わせて16の倍数
            self.rd = mujoco.Renderer(s.m, 544, 960)
            self.hc = mujoco.Renderer(s.m, 480, 640)     # 右上の小窓：頭部カメラの今の映像
            self.every = max(1, round(1 / FPS / s.m.opt.timestep))
        self.cam = mujoco.MjvCamera()
        self._aim(self.cam)
        orig = s.sim.mj_step                            # ShelfSim の1刻み（天板との接触の記録付き）を包む
        def step():
            orig(); self.tick()
        s.sim.mj_step = step

    def _aim(self, cam):
        c = self.s.c0
        cam.lookat[:] = [c[0] - 0.01, c[1], c[2] + 0.08]           # 天板と部品の両方が入るよう、ロボットの右手側の斜め前から見る
        cam.distance, cam.azimuth, cam.elevation = 0.40, 55.0, -8.0

    def markers(self, scn):
        def ball(p, r, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, [r, 0, 0], np.asarray(p, float),
                                np.eye(3).ravel(), np.asarray(rgba, np.float32))
            scn.ngeom += 1
        z = bs.TABLE_TOP_Z + RB.W + 0.004
        if self.fronts is not None:
            for x, y in self.fronts:
                ball([x, y, z], 0.0007, [1.0, 0.85, 0.1, 0.9])
        if self.touch_pt is not None:
            ball([*self.touch_pt, z + 0.006], 0.0025, [0.9, 0.15, 0.15, 1])
        if self.grasp_pt is not None:
            ball([*self.grasp_pt, z + 0.006], 0.0025, [0.15, 0.4, 0.95, 1])

    def update(self, est=None, touch=None, grasp=None, text=None):
        if est is not None:                              # 粒子ごとの本体の手前の縁（部品の軸に沿ってフランジの長さだけ奥）
            c, sn = np.cos(est[:, 2]), np.sin(est[:, 2])
            self.fronts = np.column_stack([est[:, 0] + c * est[:, 3], est[:, 1] + sn * est[:, 3]])
        if touch is not None:
            self.touch_pt = touch
        if grasp is not None:
            self.grasp_pt = grasp
        if text is not None:
            self.text = text; print(text)
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0; self.markers(self.viewer.user_scn)
            self.viewer.set_texts((None, None, self.text, ""))
            self.viewer.sync()

    def tick(self):
        self.n += 1
        if self.viewer is not None:
            if self.n % 8 == 0:                          # 16ms ごとに画面を更新し、実時間（×speed）に合わせて待つ
                if not self.viewer.is_running():
                    raise SystemExit("表示ウィンドウが閉じられました")
                self.viewer.sync()
                lag = self.n * self.s.m.opt.timestep / self.speed - (time.time() - self.t0)
                if lag > 0:
                    time.sleep(lag)
        if self.frames is not None and self.n % self.every == 0:
            self.rd.update_scene(self.s.d, self.cam); self.markers(self.rd.scene)
            img = self.rd.render()
            m = self.s.m; b = m.geom("board").id; rgba = m.geom_rgba[b].copy()
            m.geom_rgba[b] = [0.6, 0.6, 0.65, 1.0]      # 頭部カメラには天板を元どおり不透明に描く
            self.hc.update_scene(self.s.d, "head_cam"); head = self.hc.render()
            m.geom_rgba[b] = rgba
            self.frames.append(self.caption(img, head))

    def caption(self, img, head=None):
        from PIL import Image, ImageDraw
        im = Image.fromarray(img); dr = ImageDraw.Draw(im)
        if head is not None and self.head_crop is not None:   # 右上の小窓：頭部カメラの今の映像（部品のまわりを1.5倍に拡大）
            x0, y0 = self.head_crop
            small = Image.fromarray(head[y0:y0 + 160, x0:x0 + 214]).resize((320, 240))
            im.paste((255, 255, 255), (960 - 334, 10, 960 - 10, 254)); im.paste(small, (960 - 332, 12))
            dr.text((960 - 330, 258), "head camera (live): the shelf hides most of the part", fill=(255, 255, 255))
        for k, line in enumerate(self.text.split("\n")):
            dr.text((12, 10 + 16 * k), line, fill=(255, 255, 255))
        dr.text((12, 544 - 22), "red: touch point   blue: grasp point   yellow: body front edge of each hypothesis",
                fill=(230, 230, 230))
        return np.array(im)


def run(i, rule, n_touch, video=None, speed=1.0, board_alpha=0.75):
    """run_touchsel.trial と同じ手順。掴む動作だけは複製でなく本物の状態で行う（表示のため）。"""
    lay = RB.layout(i); rng = np.random.default_rng(5000 + i); rr = np.random.default_rng(9000 + i)
    s = S.ShelfSim(RB.W, visible=lay["vis"], part=(lay["fl"], lay["L"]), dxy=(0, lay["dy"]), yaw_deg=lay["yaw"])
    RB.apply_phys(s, lay); F.use_predictor("predictor.pkl", s.front_x)
    m, d = s.m, s.d
    cam, dep, cm, tm, occl = P.observe(m, d, s.cube, rng)        # カメラの観測は表示ウィンドウを開く前に済ませる
    _, ztab = P.estimate_size(cam, dep, cm, tm)
    yaw0 = P.yaw_from_depth(cam, dep, cm, ztab)
    est, _ = P.init_particles_part(cam, cm, occl, RB.W, ztab, rng, (S.X_FRONT, S.Y0), RB.FL_R, RB.L_R, yaw0=yaw0)
    est, seen = P.refine_part_depth(est, cam, dep, cm, ztab, s.front_x, bs.TABLE_TOP_Z + bs.SHELF_H, RB.FL_R, rng)
    u, v = cam.project([s.c0])[0]                                # 動画の小窓：頭部カメラの画像で部品のまわり 214×160 画素
    head_crop = (int(np.clip(u - 107, 0, 640 - 214)), int(np.clip(v - 95, 0, 480 - 160)))
    # ここから表示だけの設定（観測は済んでいるので結果に影響しない）
    m.geom_rgba[m.geom("board").id] = [0.78, 0.80, 0.86, board_alpha]   # 天板を明るい半透明に
    m.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]              # 天板の下面が真っ黒にならないよう、環境光を足す
    m.light_castshadow[:] = 0                                    # 天板の影で部品が見えにくいので、表示では影を消す

    head = (f"layout {i}  visible {lay['vis'] * 1000:.0f} mm  flange {lay['fl'] * 1000:.1f} mm  "
            f"body {lay['L'] * 1000:.1f} mm  rule {rule}")
    viewer = None
    if video is None:
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
    sh = Show(s, viewer, video, speed, head_crop)
    if viewer is not None:
        with viewer.lock():
            sh._aim(viewer.cam)
    sd = lambda e: (e[:, 0] + e[:, 3]).std() * 1000              # run_touchsel と同じ「本体の手前の縁のx」のばらつき
    sh.update(est=est, text=f"{head}\nbody seen by camera: {'yes' if seen else 'no'}   "
                            f"body front spread (sd) {sd(est):.1f} mm")
    for k in range(1, n_touch + 1):
        gc_ = F.axis_points(est); tc = F.axis_points(est, step=0.003)
        tx, ty = T.choose(rule, est, gc_, tc, rr)
        sh.update(touch=(tx, ty), text=f"{head}\ntouch {k}/{n_touch}: moving to the red point")
        r = s.touch([tx, ty])
        est = F.update(est, tx, ty, r["obs"], rng)
        sh.update(est=est, text=f"{head}\ntouch {k}/{n_touch}: {OBS_NAME[r['obs']]}   "
                                f"body front spread (sd) {sd(est):.1f} mm")
    (gx, gy), p = F.best_grasp_m(est, F.axis_points(est))
    sh.update(grasp=(gx, gy), text=f"{head}\ngrasp at the blue point   predicted success {p:.2f}")
    g = s.grasp([gx, gy, ztab + RB.W / 2])
    result = "SUCCESS" if g["success"] else "FAILURE"
    sh.update(text=f"{head}\n{result}   lifted {g['rise_cm']:.1f} cm   (predicted {p:.2f})")
    s.sim.step(1.0)                                              # 結果を1秒見せる

    ref = saved_results().get(i, {}).get(rule)
    if ref and n_touch <= len(ref["steps"]):
        same = ref["steps"][n_touch - 1]["success"] == g["success"]
        print(f"保存済みの結果（results/tsel）: {'成功' if ref['steps'][n_touch - 1]['success'] else '失敗'} → "
              f"{'一致' if same else '不一致'}")
    if video:
        import imageio.v2 as imageio
        imageio.mimsave(video, sh.frames, fps=FPS); sh.rd.close(); sh.hc.close()
        print(f"保存: {video}（{len(sh.frames)} コマ）")
    if viewer is not None:
        print("表示ウィンドウを閉じると終了します")
        while viewer.is_running():
            time.sleep(0.1)
    return g["success"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="1配置の試行を表示ウィンドウで見る、または動画に保存する")
    ap.add_argument("layout", type=int, nargs="?", default=64, help="配置の番号（0〜239）")
    ap.add_argument("--rule", default="voi", choices=T.RULES, help="触る点の選び方")
    ap.add_argument("--touches", type=int, default=1, choices=(1, 2), help="触る回数")
    ap.add_argument("--video", help="動画のファイル名（与えると表示ウィンドウを開かずに保存する）")
    ap.add_argument("--speed", type=float, default=1.0, help="再生の速さ（表示ウィンドウのとき。2なら2倍速）")
    ap.add_argument("--list", action="store_true", help="voi が成功し unc が失敗した配置を一覧する")
    a = ap.parse_args()
    if a.list:
        list_layouts()
    else:
        run(a.layout, a.rule, a.touches, a.video, a.speed)
