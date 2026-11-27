"""Build the simulation scene programmatically with MjSpec.

Using MjSpec (rather than an <include>) keeps Menagerie's relative mesh paths
working and lets us add objects, sensors and, later, soft fingertip pads in code.
"""
from pathlib import Path

import mujoco

from .pinch import PINCH_POINT
from .soft_pads import OBJECT_CONTYPE, add_soft_pads

REPO = Path(__file__).resolve().parents[1]
LEAP_XML = REPO / "third_party" / "mujoco_menagerie" / "leap_hand" / "right_hand.xml"

FINGERTIPS = ("if_ds", "mf_ds", "rf_ds", "th_ds")


def build_scene(ball_radius: float = 0.045, ball_pos=(0.05, 0.0, 0.2), soft_pads: bool = False,
                pinch: bool = False) -> mujoco.MjModel:
    """Power grasp of a ball on the palm by default; `pinch=True` gives a small sphere held
    between thumb and index. `soft_pads=True` replaces the rigid fingertips with FEM pads."""
    if not LEAP_XML.exists():
        raise FileNotFoundError(f"{LEAP_XML} not found - run scripts/fetch_models.sh first")
    spec = mujoco.MjSpec.from_file(str(LEAP_XML))

    # Lighting and floor.
    spec.worldbody.add_light(pos=[0, 0, 1], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0.5, 0.5, 0.01], rgba=[0.2, 0.25, 0.3, 1])

    if soft_pads:
        add_soft_pads(spec)

    if pinch:   # small sphere floating between thumb and index; gravity is switched on at release
        ball_radius, ball_pos, mass = 0.016, PINCH_POINT, 0.02
    else:       # dropped onto the upturned palm, then the hand closes on it
        mass = 0.05
    ball = spec.worldbody.add_body(name="ball", pos=list(ball_pos))
    ball.add_freejoint()
    ball.gravcomp = 1.0 if pinch else 0.0
    ball.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[ball_radius, 0, 0],
                  mass=mass, friction=[1.2, 0.01, 0.001], condim=4, rgba=[0.9, 0.45, 0.2, 1],
                  contype=OBJECT_CONTYPE, conaffinity=OBJECT_CONTYPE)

    # Contact visualisation scale (force arrows in N -> metres).
    spec.visual.map.force = 0.0002 if soft_pads else 0.0005
    spec.visual.scale.contactwidth = 0.05
    spec.visual.scale.contactheight = 0.01
    spec.visual.scale.forcewidth = 0.01
    return spec.compile()
