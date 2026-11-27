"""Contact extraction: per-contact points/forces and per-body summaries."""
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Contact:
    body: str          # hand body touching the object
    pos: np.ndarray    # world-space contact point
    normal: np.ndarray # world-space contact normal (from geom1 to geom2)
    force: np.ndarray  # world-space contact force on the object [N]


def object_contacts(model: mujoco.MjModel, data: mujoco.MjData, obj_body: str) -> list[Contact]:
    obj_id = model.body(obj_body).id
    out, f = [], np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
        if obj_id not in (b1, b2):
            continue
        mujoco.mj_contactForce(model, data, i, f)
        frame = c.frame.reshape(3, 3)          # rows: normal, tangent1, tangent2
        f_world = frame.T @ f[:3]               # force on geom2 expressed in world frame
        sign = 1.0 if b2 == obj_id else -1.0    # make it the force *on the object*
        other = b1 if b2 == obj_id else b2
        out.append(Contact(model.body(other).name, c.pos.copy(), frame[0].copy(), sign * f_world))
    return out


def per_body_normal_force(contacts: list[Contact]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for c in contacts:
        totals[c.body] = totals.get(c.body, 0.0) + abs(float(c.force @ c.normal))
    return totals
