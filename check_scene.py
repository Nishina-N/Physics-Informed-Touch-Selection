"""シーンの動作確認：立たせて2秒シミュレーションし、頭部カメラの画像・領域分割・深度を保存する。"""
import numpy as np
import mujoco
from PIL import Image
from build_scene import build

model = build().compile()
data = mujoco.MjData(model)

# 立ち姿勢のキーフレーム（menagerie付属'stand'）を関節だけ反映
key = model.key("stand")
nq_robot = model.nq - 7 * 2          # 立方体2個のフリージョイント分を除く
data.qpos[:nq_robot] = key.qpos[:nq_robot]
data.ctrl[:] = key.ctrl
mujoco.mj_forward(model, data)

for _ in range(int(2.0 / model.opt.timestep)):
    mujoco.mj_step(model, data)

for name in ["cube_50mm", "cube_20mm"]:
    print(name, "位置", np.round(data.body(name).xpos, 3))

W, H = 640, 480
r = mujoco.Renderer(model, H, W)
r.update_scene(data, camera="head_cam")
rgb = r.render()
r.enable_depth_rendering(); r.update_scene(data, camera="head_cam"); depth = r.render(); r.disable_depth_rendering()
r.enable_segmentation_rendering(); r.update_scene(data, camera="head_cam"); seg = r.render(); r.disable_segmentation_rendering()

Image.fromarray(rgb).save("head_rgb.png")
d = np.clip(depth, 0, 2.0) / 2.0
Image.fromarray((255 * (1 - d)).astype(np.uint8)).save("head_depth.png")
for name in ["cube_50mm", "cube_20mm"]:
    gid = model.geom(name).id
    mask = (seg[..., 0] == gid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
    print(name, "見えている画素数", int(mask.sum()),
          "平均深度[m]", round(float(depth[mask].mean()), 3) if mask.any() else None)
print("保存：head_rgb.png, head_depth.png")