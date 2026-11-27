"""Thumb-index pinch: object floats (gravity-compensated) while the hand closes, then gravity is
switched on and the pads must hold it by friction."""
import numpy as np

PINCH_POSE = {"if_mcp_act": 1.4, "if_pip_act": 0.2, "if_dip_act": 0.12,
              "th_cmc_act": 1.2, "th_axl_act": 1.0, "th_mcp_act": 0.3, "th_ipl_act": 0.3}
PINCH_POINT = (-0.033, -0.008, 0.241)   # midpoint between index and thumb pads at PINCH_POSE


def pinch_target(model, squeeze=0.1):
    extra = {"if_mcp_act": squeeze, "th_cmc_act": squeeze}
    tgt = np.array([PINCH_POSE.get(model.actuator(i).name, 0.0) + extra.get(model.actuator(i).name, 0.0)
                    for i in range(model.nu)])
    return np.clip(tgt, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def pinch_ctrl(t, target, close=2.0):
    return np.clip(t / close, 0.0, 1.0) * target


def gravity_on(t, release=2.5):
    return t >= release
