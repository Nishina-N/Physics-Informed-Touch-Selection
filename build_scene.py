"""G1（Dex3相当ハンド付き）＋机＋立方体＋頭部カメラのシーンを作る。"""
import math
import mujoco

MENAGERIE = "third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml"

# 頭部カメラ（D435i）の取り付け位置。unitree_rosのG1 URDFのd435_joint（torso_link基準）を想定。要確認
D435_POS = [0.0576235, 0.01753, 0.42987]
D435_PITCH = 0.8307767  # rad（下向き約48度）

TABLE_TOP_Z = 0.80            # 机の天板の高さ[m]（仮。手が届く高さに後で調整）
TABLE_CENTER = [0.45, -0.15]  # 机の中心（ロボットの前方・右手側）
DENSITY = 600.0               # 木材相当 [kg/m^3]


def build(cube_sizes=(0.05, 0.02)):
    spec = mujoco.MjSpec.from_file(MENAGERIE)

    # 骨盤を世界に固定（フリージョイントを削除）
    pelvis = spec.body("pelvis")
    for j in list(pelvis.joints):
        spec.delete(j)

    # 頭部カメラ
    torso = spec.body("torso_link")
    d435 = torso.add_body(name="d435_link", pos=D435_POS,
                          quat=[math.cos(D435_PITCH / 2), 0, math.sin(D435_PITCH / 2), 0])
    # URDFのリンク（x前方）→ MuJoCoのカメラ（-z方向を見る、yが上）
    d435.add_camera(name="head_cam", pos=[0, 0, 0], xyaxes=[0, -1, 0, 0, 0, 1], fovy=58)

    # 机
    world = spec.worldbody
    hz = TABLE_TOP_Z / 2
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.30, 0.40, hz], pos=[TABLE_CENTER[0], TABLE_CENTER[1], hz],
                   rgba=[0.55, 0.45, 0.35, 1])

    # 立方体（自由に動く）
    for i, s in enumerate(cube_sizes):
        name = f"cube_{int(s * 1000)}mm"
        b = world.add_body(name=name, pos=[TABLE_CENTER[0] - 0.05, TABLE_CENTER[1] + 0.10 * i, TABLE_TOP_Z + s / 2])
        b.add_freejoint(name=name + "_free")
        b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2] * 3,
                   density=DENSITY, rgba=[0.2, 0.6, 0.9, 1] if i == 0 else [0.9, 0.5, 0.2, 1],
                   friction=[0.5, 0.005, 0.0001])
    return spec


if __name__ == "__main__":
    model = build().compile()
    print("OK: nq", model.nq, "nu", model.nu, "ncam", model.ncam)