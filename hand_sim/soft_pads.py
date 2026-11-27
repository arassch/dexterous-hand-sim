"""Soft FEM fingertip pads built with MuJoCo flex.

Each pad is a volumetric tetrahedral ellipsoid attached to a distal phalanx.
Its inner core is pinned to the link (the "bone"); the outer layer is free
elastic tissue (linear FEM elasticity in MuJoCo's flex). The pad replaces the
fingertip's rigid collision geom and only collides with objects (contype bit 2),
not with the hand itself.
"""
import mujoco
import numpy as np

from .padmesh import ellipsoid_tets

PAD_RADII = np.array([0.0115, 0.016, 0.0125]) * 1.3   # pad half-extents in the distal-link frame [m]
PAD_CENTER = {                                  # pad centre in each distal-link frame [m]
    "if": [-0.0013, -0.0346, 0.0145],
    "mf": [-0.0013, -0.0346, 0.0145],
    "rf": [-0.0013, -0.0346, 0.0145],
    "th": [-0.0013, -0.0466, -0.0145],
}
PAD_CONTYPE = 2          # pads collide with objects that have this bit, not with the hand (bit 1)
OBJECT_CONTYPE = 1 | 2   # objects collide with both hand and pads


def _pad_xml(young, poisson, damping, mass, resolution, pin_radius):
    P, T = ellipsoid_tets(PAD_RADII, resolution)
    pin = np.where(np.linalg.norm(P / PAD_RADII, axis=1) < pin_radius)[0]
    return f"""<mujoco><worldbody><body name="pad">
  <flexcomp name="flex" type="direct" dim="3" radius="0.0005" rgba=".95 .78 .72 1" mass="{mass}"
    point="{' '.join(f'{v:.6f}' for v in P.ravel())}" element="{' '.join(map(str, T.ravel()))}">
    <edge equality="false"/>
    <elasticity young="{young}" poisson="{poisson}" damping="{damping}"/>
    <contact condim="3" selfcollide="none" contype="{PAD_CONTYPE}" conaffinity="{PAD_CONTYPE}" friction="1.2"/>
    <pin id="{' '.join(map(str, pin))}"/>
  </flexcomp></body></worldbody></mujoco>"""


def add_soft_pads(spec: mujoco.MjSpec, fingers=("if", "mf", "rf", "th"), young=1e5, poisson=0.4,
                  damping=5e-4, mass=0.01, resolution=5, pin_radius=0.7, shell_alpha=0.35):
    """Attach FEM pads to the LEAP distal links. Returns the pad flex names in order."""
    # Flex elasticity needs the discrete (or explicit Euler) integrator in MuJoCo 4.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    spec.option.timestep = 0.002
    xml = _pad_xml(young, poisson, damping, mass, resolution, pin_radius)
    names = []
    for f in fingers:
        tip = spec.geom(f"{f}_tip")
        tip.contype = tip.conaffinity = 0                     # the pad replaces the rigid tip
        if shell_alpha < 1:                                   # see-through white tip shell so the pad is visible
            tip.material = ""
            tip.rgba = [0.95, 0.95, 0.95, shell_alpha]
        frame = spec.body(f"{f}_ds").add_frame(pos=PAD_CENTER[f])
        frame.attach_body(mujoco.MjSpec.from_string(xml).body("pad"), f"{f}_", "")
        names.append(f"{f}_flex")
    return names
