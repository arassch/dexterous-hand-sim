"""Contact pressure on the soft fingertip pads.

MuJoCo reports flex-object contacts per tetrahedron. Each contact's normal force is
spread over that tet's nodes that lie on the pad's outer (palmar) face, weighted by
inverse distance to the contact point. Dividing by the area each surface node
represents (a quarter of each adjacent quad) gives a nodal pressure field.

The pad's outer face is the grid face x = -1 of the pad mesh, so with resolution n
each pad reads out as an n x n taxel image (rows: along the finger, base -> tip end;
columns: across the finger).
"""
import mujoco
import numpy as np

from .padmesh import pad_tets
from .soft_pads import PAD_BOXINESS, PAD_CURL, PAD_RADII


class PadPressure:
    def __init__(self, model: mujoco.MjModel, resolution: int = 5):
        self.n = resolution
        P, _, G = pad_tets(PAD_RADII, resolution, boxiness=PAD_BOXINESS, **PAD_CURL)
        # local ids of outer-face nodes, laid out as an n x n grid [along finger, across finger]
        outer = np.where(np.isclose(G[:, 0], -1.0))[0]
        order = np.lexsort((G[outer, 2], -G[outer, 1]))         # base (+y) -> tip end (-y), then z
        self.face = outer[order].reshape(resolution, resolution)
        self.is_face = np.zeros(len(P), bool)
        self.is_face[self.face.ravel()] = True
        self.pads = {}
        for i in range(model.nflex):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, i)
            if model.flex_vertnum[i] != len(P) or model.flex_dim[i] != 3:
                continue
            self.pads[name] = i
        self.force = {k: np.zeros(len(P)) for k in self.pads}
        self.pressure = {k: np.zeros(len(P)) for k in self.pads}

    def _face_areas(self, data, fid):
        """Area represented by each outer-face node (quarter of each adjacent grid quad)."""
        model_vert0 = self._vertadr[fid]
        X = data.flexvert_xpos[model_vert0 + self.face]           # n x n x 3
        area = np.zeros(self.face.shape)
        for a in range(self.n - 1):
            for b in range(self.n - 1):
                p00, p01, p10, p11 = X[a, b], X[a, b + 1], X[a + 1, b], X[a + 1, b + 1]
                q = 0.5 * (np.linalg.norm(np.cross(p01 - p00, p10 - p00)) +
                           np.linalg.norm(np.cross(p01 - p11, p10 - p11)))
                for (u, v) in ((a, b), (a, b + 1), (a + 1, b), (a + 1, b + 1)):
                    area[u, v] += q / 4
        return area

    def update(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._vertadr = model.flex_vertadr
        for k in self.force:
            self.force[k][:] = 0
        inv = {fid: name for name, fid in self.pads.items()}
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            side = 0 if c.flex[0] in inv else 1 if c.flex[1] in inv else -1
            if side < 0 or c.flex[1 - side] >= 0:               # pad vs rigid geom only
                continue
            fid, elem = c.flex[side], c.elem[side]
            mujoco.mj_contactForce(model, data, i, f6)
            adr = model.flex_elemdataadr[fid] + elem * (model.flex_dim[fid] + 1)
            verts = model.flex_elem[adr:adr + model.flex_dim[fid] + 1]
            verts = [v for v in verts if self.is_face[v]] or list(verts)
            X = data.flexvert_xpos[model.flex_vertadr[fid] + np.array(verts)]
            w = 1.0 / (np.linalg.norm(X - c.pos, axis=1) + 1e-4)
            self.force[inv[fid]][verts] += abs(f6[0]) * w / w.sum()
        for name, fid in self.pads.items():
            area = self._face_areas(data, fid)
            p = np.zeros_like(self.force[name])
            p[self.face] = self.force[name][self.face] / np.maximum(area, 1e-9)
            self.pressure[name] = p

    def taxels(self, name: str) -> np.ndarray:
        """n x n pressure image [Pa] for a pad."""
        return self.pressure[name][self.face]

    def face_positions(self, model, data, name):
        fid = self.pads[name]
        return data.flexvert_xpos[model.flex_vertadr[fid] + self.face]
