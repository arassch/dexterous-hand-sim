"""Objects for the pinch, posed relative to the pinch axis.

Frame used to place objects at the pinch point:
  e1 = pinch axis (index pad -> thumb pad), e3 ~ along the fingers (vertical at the pinch pose),
  e2 = across the fingers. On the tactile images e3 is the row direction and e2 the column one.
Every object is 38 mm across along e1, so it just fits between the pads.
"""
import mujoco
import numpy as np

from .pinch import PINCH_POINT

PINCH_AXIS = np.array([-0.9972, -0.0484, 0.0575])   # index -> thumb pad at PINCH_POSE
HALF_GAP = 0.019                                     # half the pad-to-pad gap at first contact [m]
OBJECTS = ("sphere", "cube", "cylinder", "plate")


def _basis():
    e1 = PINCH_AXIS / np.linalg.norm(PINCH_AXIS)
    e2 = np.cross([0.0, 0.0, 1.0], e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    return e1, e2, e3


def _quat(x, y, z):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def add_object(spec: mujoco.MjSpec, kind: str, contype: int, mass: float = 0.02):
    """Add a free object named 'object' at the pinch point. Returns the body spec."""
    e1, e2, e3 = _basis()
    G = mujoco.mjtGeom
    if kind == "sphere":
        gtype, size, quat = G.mjGEOM_SPHERE, [HALF_GAP, 0, 0], _quat(e1, e2, e3)
    elif kind == "cube":       # flat faces on both pads; 20 mm cross-section so its edges land on the pad
        gtype, size, quat = G.mjGEOM_BOX, [HALF_GAP, 0.010, 0.010], _quat(e1, e2, e3)
    elif kind == "cylinder":   # pinched across its axis, axis tilted 30 deg from vertical in the pad plane
        a = np.cos(np.radians(30)) * e3 + np.sin(np.radians(30)) * e2
        gtype, size, quat = G.mjGEOM_CYLINDER, [HALF_GAP, 0.022, 0], _quat(e1, np.cross(a, e1), a)
    elif kind == "plate":      # 2 mm plate held edge-on: a thin line of contact, diagonal across the pad
        y = np.cos(np.radians(45)) * e3 + np.sin(np.radians(45)) * e2
        gtype, size, quat = G.mjGEOM_BOX, [HALF_GAP, 0.014, 0.001], _quat(e1, y, np.cross(e1, y))
    else:
        raise ValueError(f"unknown object {kind!r}; choose from {OBJECTS}")
    body = spec.worldbody.add_body(name="object", pos=list(PINCH_POINT), quat=list(quat))
    body.add_freejoint()
    body.gravcomp = 1.0          # floats until the pinch controller switches gravity on
    body.add_geom(name="object", type=gtype, size=size, mass=mass, friction=[1.2, 0.01, 0.001],
                  condim=4, rgba=[0.9, 0.45, 0.2, 1], contype=contype, conaffinity=contype)
    return body
