"""Thumb-index pinch: object floats (gravity-compensated) while the hand closes, then gravity is
switched on and the pads must hold it by friction."""
import numpy as np

# Pose where the index and thumb palmar pads face each other (found by sampling joint space).
PINCH_POSE = {"if_mcp_act": 1.06, "if_rot_act": 0.16, "if_pip_act": 0.36, "if_dip_act": 0.40,
              "th_cmc_act": 1.49, "th_axl_act": 0.10, "th_mcp_act": 0.31, "th_ipl_act": 0.13}
PINCH_POINT = (-0.017, -0.001, 0.232)   # midpoint between index and thumb pads at PINCH_POSE
PINCH_RADIUS = 0.017                     # sphere radius that just fits between the pads


def pinch_target(model, squeeze=0.1):
    extra = {"if_pip_act": squeeze, "if_dip_act": squeeze, "th_mcp_act": squeeze, "th_ipl_act": squeeze}
    tgt = np.array([PINCH_POSE.get(model.actuator(i).name, 0.0) + extra.get(model.actuator(i).name, 0.0)
                    for i in range(model.nu)])
    return np.clip(tgt, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def pinch_ctrl(t, target, close=2.0):
    return np.clip(t / close, 0.0, 1.0) * target


def gravity_on(t, release=2.5):
    return t >= release
