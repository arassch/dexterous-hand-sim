"""Contact extraction: per-contact points/forces and per-body/pad summaries."""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Contact:
    body: str          # hand body or soft-pad flex touching the object
    pos: np.ndarray    # world-space contact point
    normal: np.ndarray # world-space contact normal (from side 0 to side 1)
    force: np.ndarray  # world-space contact force on the object [N]
    vert: int = -1     # pad vertex index for flex contacts, else -1


def _side_name(model, c, k):
    if c.flex[k] >= 0:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, c.flex[k])
    return model.body(model.geom_bodyid[c.geom[k]]).name


def object_contacts(model: mujoco.MjModel, data: mujoco.MjData, obj_body: str) -> list[Contact]:
    obj_id = model.body(obj_body).id
    out, f = [], np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        sides = [model.geom_bodyid[c.geom[k]] if c.geom[k] >= 0 else -1 for k in (0, 1)]
        if obj_id not in sides:
            continue
        k_obj = sides.index(obj_id)
        k_hand = 1 - k_obj
        mujoco.mj_contactForce(model, data, i, f)
        frame = c.frame.reshape(3, 3)            # rows: normal, tangent1, tangent2
        f_world = frame.T @ f[:3]                 # force on side 1, world frame
        sign = 1.0 if k_obj == 1 else -1.0        # make it the force *on the object*
        out.append(Contact(_side_name(model, c, k_hand), c.pos.copy(), frame[0].copy(),
                           sign * f_world, int(c.vert[k_hand]) if c.flex[k_hand] >= 0 else -1))
    return out


def per_body_normal_force(contacts: list[Contact]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for c in contacts:
        totals[c.body] = totals.get(c.body, 0.0) + abs(float(c.force @ c.normal))
    return totals


def per_body_count(contacts: list[Contact]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in contacts:
        counts[c.body] = counts.get(c.body, 0) + 1
    return counts
