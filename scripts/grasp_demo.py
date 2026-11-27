"""Scripted LEAP-hand grasp with contact points and forces.

Interactive (macOS needs mjpython for the passive viewer):
    mjpython scripts/grasp_demo.py
Headless (prints per-finger normal forces):
    python scripts/grasp_demo.py --headless
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_sim.contacts import object_contacts, per_body_normal_force  # noqa: E402
from hand_sim.grasp import ctrl_at, grasp_target  # noqa: E402
from hand_sim.scene import build_scene  # noqa: E402


def run_headless(model, data, duration):
    target = grasp_target(model)
    while data.time < duration:
        data.ctrl[:] = ctrl_at(data.time, target)
        mujoco.mj_step(model, data)
    contacts = object_contacts(model, data, "object")
    print(f"t={data.time:.2f}s  ball at {np.round(data.body('object').xpos, 3)}  contacts={len(contacts)}")
    for body, fn in sorted(per_body_normal_force(contacts).items()):
        print(f"  {body:8s} {fn:6.2f} N")


def run_viewer(model, data, duration):
    import mujoco.viewer
    target = grasp_target(model)
    with mujoco.viewer.launch_passive(model, data) as v:
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            v.cam.lookat[:] = [0.0, 0.0, 0.15]
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.45, 135, -25
        while v.is_running():
            start = time.time()
            data.ctrl[:] = ctrl_at(data.time % duration, target)
            if data.time % duration < model.opt.timestep:   # loop the demo
                mujoco.mj_resetData(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=8.0)
    args = ap.parse_args()
    model = build_scene()
    data = mujoco.MjData(model)
    (run_headless if args.headless else run_viewer)(model, data, args.duration)
