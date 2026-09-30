"""G1（Dex3相当ハンド付き）＋机＋立方体＋頭部カメラのシーンを作る。"""
import math
import mujoco

MENAGERIE = "third_party/mujoco_menagerie/unitree_g1/scene_with_hands.xml"

# 頭部カメラ（D435i）の取り付け位置。unitree_ros g1_29dof_with_hand.urdf の d435_joint（z=0.41987）に、
# menagerieのtorso_link原点がURDFより10mm低い分（肩関節 0.24778 vs 0.23778 で確認）を足した値
D435_POS = [0.0576235, 0.01753, 0.42987]
D435_PITCH = 0.8307767  # rad（下向き約48度）

TABLE_TOP_Z = 0.80      # 机の天板の高さ[m]（仮。手が届く高さに後で調整）
TABLE_CENTER = [0.45, -0.15]  # 机の中心（ロボットの前方・右手側）
DENSITY = 600.0
IMPRATIO = 10.0        # 摩擦方向の拘束の硬さ（既定1）。大きいほど接触面で滑りにくい
CUBE_CONDIM = 4        # 立方体の接触の次元（4＝ねじり摩擦あり）


FLANGE_H = 0.004       # 部品の手前の低いフランジの高さ[m]


def build(cube_sizes=(0.05, 0.02), lengths=None, parts=None):
    """lengths: 立方体ごとの奥行き方向の長さ[m]（Noneなら立方体）。名前は幅で付ける（cube_15mm など）。
    parts: 立方体ごとに None か (flange_len, body_len)。与えると、手前に高さ FLANGE_H・長さ flange_len の
    フランジ（名前 <name>_flange）、その奥に高さ＝幅・長さ body_len の本体（名前 <name>）をつないだ1つの剛体にする。
    剛体の原点は本体の中心。"""
    spec = mujoco.MjSpec.from_file(MENAGERIE)

    # 把持での滑りを抑える接触設定（楕円の摩擦錐＋impratio。MuJoCoで把持を扱うときの定番）
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = IMPRATIO

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
        L = s if lengths is None or lengths[i] is None else lengths[i]
        part = None if parts is None else parts[i]
        if part is not None:
            L = part[1]
        rgba = [0.2, 0.6, 0.9, 1] if i == 0 else [0.9, 0.5, 0.2, 1]
        b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[L / 2, s / 2, s / 2],
                   density=DENSITY, rgba=rgba,
                   friction=[0.5, 0.005, 0.0001],
                   condim=CUBE_CONDIM)   # ねじり摩擦あり（柔らかい指先パッドは挟む軸まわりの回転に抵抗する。既定の3では0）
        if part is not None:
            fl = part[0] + 0.001                        # 本体と1mm重ねてつなぐ
            b.add_geom(name=name + "_flange", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[fl / 2, s / 2, FLANGE_H / 2],
                       pos=[-L / 2 - part[0] + fl / 2, 0, -s / 2 + FLANGE_H / 2],
                       density=DENSITY, rgba=rgba, friction=[0.5, 0.005, 0.0001], condim=CUBE_CONDIM)
    return spec


# 棚：机上 SHELF_H の高さに天板を置き、その前端を物体の手前に合わせる（前端からの奥行きで見え率を変える）
SHELF_H = 0.20          # 机上から天板下面まで[m]（18cmでは触るときに手首が当たったので20cmに上げた）
SHELF_T = 0.015         # 天板の厚さ[m]
SHELF_DEPTH = 0.40      # 天板の奥行き方向の長さ[m]
SHELF_WIDTH = 0.60      # 天板の左右の長さ[m]


def add_shelf(spec, front_x, center_y):
    """天板を追加する。front_x は天板の前端（ロボット側の縁）のx座標。"""
    spec.worldbody.add_geom(
        name="board", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[SHELF_DEPTH / 2, SHELF_WIDTH / 2, SHELF_T / 2],
        pos=[front_x + SHELF_DEPTH / 2, center_y, TABLE_TOP_Z + SHELF_H + SHELF_T / 2],
        rgba=[0.6, 0.6, 0.65, 1])
    return spec


def build_shelf(cube_size, front_x, center_y, spare_size=0.05, length=None, part=None):
    """机＋棚＋物体（使う1個と、机の端へ退避させる1個）。length を与えると奥行き方向の長さがそれの直方体。
    part=(flange_len, body_len) を与えるとフランジ付きの部品。"""
    return add_shelf(build(cube_sizes=(spare_size, cube_size), lengths=(None, length),
                           parts=(None, part)), front_x, center_y)


if __name__ == "__main__":
    spec = build()
    model = spec.compile()
    with open("scene_g1_cubes.xml", "w") as f:
        f.write(spec.to_xml())
    print("OK: nq", model.nq, "nu", model.nu, "ncam", model.ncam)
