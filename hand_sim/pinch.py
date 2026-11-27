"""Thumb-index pinch with opposing straight-line approach.

The hand first moves to a pre-grasp where the index and thumb pads face each other
~2 cm apart from the object. Each fingertip then closes straight along its pad normal
(joint direction from the pad-point Jacobian), so the thumb and index approach from
opposite sides, and finally squeezes a little past contact. The object floats
(gravity-compensated) until `release`, then the pads must hold it by friction.
"""
import mujoco
import numpy as np

from .soft_pads import PAD_CENTER

# Pose where the index and thumb palmar pads directly oppose each other (normals at 169 deg, each
# within 6 deg of the line between them). Found by sampling joint space, then local refinement
# that also rewards being able to move each pad straight along that line.
PINCH_POSE = {"if_mcp": 1.44, "if_rot": 0.03, "if_pip": 0.33, "if_dip": -0.16,
              "th_cmc": 1.49, "th_axl": 0.06, "th_mcp": -0.03, "th_ipl": 0.18}
FINGER_JOINTS = {"if": ["if_mcp", "if_rot", "if_pip", "if_dip"],
                 "th": ["th_cmc", "th_axl", "th_mcp", "th_ipl"]}
PINCH_POINT = (-0.043, -0.007, 0.243)   # midpoint between index and thumb pads at PINCH_POSE

T_PREGRASP, T_APPROACH, T_RELEASE = 1.5, 1.5, 3.5   # seconds


def _pad_point(model, data, f):
    b = data.body(f + "_ds")
    R = b.xmat.reshape(3, 3)
    return b.xpos + R @ np.array(PAD_CENTER[f]), -R[:, 0], b.id   # point, outward palmar normal, body


class PinchController:
    def __init__(self, model, approach=0.02, squeeze=0.008):
        self.model = model
        d = mujoco.MjData(model)
        self.adr = {j: model.jnt_qposadr[model.joint(j).id] for js in FINGER_JOINTS.values() for j in js}
        for j, v in PINCH_POSE.items():
            d.qpos[self.adr[j]] = v
        mujoco.mj_forward(model, d)
        p_if, _, _ = _pad_point(model, d, "if")
        p_th, _, _ = _pad_point(model, d, "th")
        u = (p_th - p_if) / np.linalg.norm(p_th - p_if)   # index -> thumb
        self.dq = {}                                        # joint motion per metre of pad travel
        for f, direction in (("if", u), ("th", -u)):        # pads move towards each other
            p, _, bid = _pad_point(model, d, f)
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, d, jacp, None, p, bid)
            cols = [model.jnt_dofadr[model.joint(j).id] for j in FINGER_JOINTS[f]]
            self.dq[f] = np.linalg.pinv(jacp[:, cols]) @ direction
        self.approach, self.squeeze = approach, squeeze
        self.act = {model.actuator(i).name: i for i in range(model.nu)}

    def joint_targets(self, t):
        if t < T_PREGRASP:
            a, s = t / T_PREGRASP, -self.approach
        else:
            a = 1.0
            s = -self.approach + min(1.0, (t - T_PREGRASP) / T_APPROACH) * (self.approach + self.squeeze)
        q = {}
        for f, js in FINGER_JOINTS.items():
            for j, dqj in zip(js, self.dq[f]):
                q[j] = a * (PINCH_POSE[j] + s * dqj)
        return q

    def ctrl(self, t):
        u = np.zeros(self.model.nu)
        for j, v in self.joint_targets(t).items():
            u[self.act[j + "_act"]] = v
        return np.clip(u, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])


def gravity_on(t, release=T_RELEASE):
    return t >= release
