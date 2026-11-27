"""Soft FEM fingertip pads built with MuJoCo flex.

Each pad is a volumetric tetrahedral rounded box covering the whole palmar face
(-x in the distal-link frame, the side the finger curls towards) of a distal phalanx,
curled over the end of the fingertip, like the pulp of a real finger. Its back half is
pinned to the link; the front half is free elastic tissue (FEM elasticity in MuJoCo's flex). The pad replaces the
fingertip's rigid collision geom and only collides with objects (contype bit 2),
not with the hand itself.
"""
import mujoco
import numpy as np

from .padmesh import pad_tets

# Pad half-extents in the distal-link frame [m]: thickness (x, palmar normal), along the finger (y),
# lateral (z). Sized to LEAP's tip: palmar face ~31 mm long x 24 mm wide.
PAD_RADII = np.array([0.005, 0.017, 0.0125])
PAD_BOXINESS = 0.6                     # 0 = ellipsoid, 1 = box
PAD_CURL = dict(curl_axis=1, curl_dir=-1, curl_start=0.009, curl_radius=0.010, curl_towards=0)
# Pad centre in each distal-link frame [m]. LEAP's palmar tip face is at x = -0.0113;
# centring the pad at x = -0.0088 makes it stand ~2.5 mm proud of the shell.
PAD_CENTER = {
    "if": [-0.0088, -0.0348, 0.0145],
    "mf": [-0.0088, -0.0348, 0.0145],
    "rf": [-0.0088, -0.0348, 0.0145],
    "th": [-0.0088, -0.0468, -0.0145],
}
PAD_CONTYPE = 2          # pads collide with objects that have this bit, not with the hand (bit 1)
OBJECT_CONTYPE = 1 | 2   # objects collide with both hand and pads


def _pad_xml(young, poisson, damping, mass, resolution, pin_below):
    P, T, G = pad_tets(PAD_RADII, resolution, boxiness=PAD_BOXINESS, **PAD_CURL)
    pin = np.where(G[:, 0] >= -pin_below)[0]                    # pin the back (bone-side, +x) half
    return f"""<mujoco><worldbody><body name="pad">
  <flexcomp name="flex" type="direct" dim="3" radius="0.0005" rgba=".78 .78 .78 1" mass="{mass}"
    point="{' '.join(f'{v:.6f}' for v in P.ravel())}" element="{' '.join(map(str, T.ravel()))}">
    <edge equality="false"/>
    <elasticity young="{young}" poisson="{poisson}" damping="{damping}"/>
    <contact condim="3" selfcollide="none" contype="{PAD_CONTYPE}" conaffinity="{PAD_CONTYPE}" friction="1.2"/>
    <pin id="{' '.join(map(str, pin))}"/>
  </flexcomp></body></worldbody></mujoco>"""


def add_soft_pads(spec: mujoco.MjSpec, fingers=("if", "mf", "rf", "th"), young=2e4, poisson=0.4,
                  damping=5e-4, mass=0.015, resolution=5, pin_below=0.0, shell_alpha=1.0):
    """Attach FEM pads to the LEAP distal links. Returns the pad flex names in order."""
    # Flex elasticity needs the discrete (or explicit Euler) integrator in MuJoCo 4.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
    spec.option.timestep = 0.002
    xml = _pad_xml(young, poisson, damping, mass, resolution, pin_below)
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
