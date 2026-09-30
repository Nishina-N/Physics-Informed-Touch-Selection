"""grasp_test.Sim.ik を mink で置き換える（同じ引数・戻り値）。
挟む点は手首リンクに固定なので、目標の手首姿勢 = (RT, p_target - RT @ grasp_pt) として FrameTask に渡す。
右腕7関節以外は DofFreezingTask で固定、関節範囲は ConfigurationLimit。"""
import numpy as np, mujoco, mink
import grasp_test as g

_cache = {}

def _setup(sim):
    key = id(sim.m)
    if key not in _cache:
        _cache.clear()      # 試行ごとにモデルを作り直すので、古いモデルを持ち続けない（メモリが溢れる）
        m = sim.m
        task = mink.FrameTask("right_wrist_yaw_link", "body", position_cost=1.0, orientation_cost=1.0, lm_damping=1e-3)
        frozen = [v for v in range(m.nv) if v not in sim.arm_v]
        freeze = mink.DofFreezingTask(m, dof_indices=frozen)
        _cache[key] = (mink.Configuration(m), task, freeze, [mink.ConfigurationLimit(m)])
    return _cache[key]

def ik(self, p_target, iters=300, dt=0.05):
    cfg, task, freeze, limits = _setup(self)
    q0 = self.d.qpos.copy()
    m = self.m                                   # 固定する関節（左腕など）の範囲外れで QP が不能にならないよう、コピーだけ範囲内に収める
    for j in range(m.njnt):
        if m.jnt_limited[j] and m.jnt_type[j] in (2, 3):
            a = m.jnt_qposadr[j]; q0[a] = np.clip(q0[a], *m.jnt_range[j])
    cfg.update(q0)
    RT = g.r_target()
    p_w = np.asarray(p_target) - RT @ self.grasp_pt
    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(RT), p_w))
    for _ in range(iters):
        v = mink.solve_ik(cfg, [task], dt, "daqp", damping=1e-3, limits=limits, constraints=[freeze])
        cfg.integrate_inplace(v, dt)
        e = task.compute_error(cfg)                 # [位置3, 回転3]
        if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1e-2:
            break
    R = cfg.data.xmat[self.m.body("right_wrist_yaw_link").id].reshape(3, 3)
    p = cfg.data.xpos[self.m.body("right_wrist_yaw_link").id] + R @ self.grasp_pt
    return np.array([cfg.q[a] for a in self.arm_q]), np.linalg.norm(np.asarray(p_target) - p)

def use_mink():
    g.Sim.ik_dls = g.Sim.ik
    g.Sim.ik = ik
