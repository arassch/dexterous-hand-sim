"""Tactile visualisation: pressure heat dots on the pads and n x n taxel images."""
import mujoco
import numpy as np
from matplotlib import colormaps

_CMAP = colormaps["inferno"]


def colorize(p, pmax):
    """Pressure [Pa] -> RGBA in [0,1] (inferno)."""
    return np.asarray(_CMAP(np.clip(np.asarray(p) / pmax, 0, 1)))


def taxel_image(taxels, pmax, size=160, border=2):
    """Bilinearly upsampled taxel image as uint8 RGB (size x size), base of finger at top."""
    t = np.clip(np.asarray(taxels, float) / pmax, 0, 1)
    n = t.shape[0]
    xs = np.linspace(0, n - 1, size)
    i0 = np.clip(np.floor(xs).astype(int), 0, n - 2)
    f = xs - i0
    rows = t[i0] * (1 - f)[:, None] + t[i0 + 1] * f[:, None]                    # size x n
    img = rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]        # size x size
    rgb = (_CMAP(img)[..., :3] * 255).astype(np.uint8)
    if border:
        rgb[:border], rgb[-border:], rgb[:, :border], rgb[:, -border:] = 255, 255, 255, 255
    return rgb


def add_pad_heat(scn: mujoco.MjvScene, positions, pressures, pmax, radius=0.0013, min_frac=0.03):
    """Append small spheres at the pad's outer-face nodes, coloured by pressure."""
    for x, p in zip(np.asarray(positions).reshape(-1, 3), np.asarray(pressures).ravel()):
        if p < min_frac * pmax or scn.ngeom >= scn.maxgeom:
            continue
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), x,
                            np.eye(3).ravel(), colorize(p, pmax).astype(np.float32))
        scn.ngeom += 1
