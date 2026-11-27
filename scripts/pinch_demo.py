"""Thumb-index pinch with soft FEM fingertip pads.

The hand moves to a pre-grasp, then the index and thumb close straight towards each
other along their pad normals. The sphere floats until t=3.5 s, when gravity is
switched on and the pads must hold it by friction. Contact points and forces are shown in the viewer.

Interactive (macOS needs mjpython):
    mjpython scripts/pinch_demo.py
    mjpython scripts/pinch_demo.py --rigid      # same pinch with the original rigid tips, for comparison
Headless:
    python scripts/pinch_demo.py --headless
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_sim.contacts import object_contacts, per_body_count, per_body_normal_force  # noqa: E402
from hand_sim.pinch import PinchController, gravity_on  # noqa: E402
from hand_sim.scene import build_scene  # noqa: E402


def step(model, data, controller, ball_id):
    data.ctrl[:] = controller.ctrl(data.time)
    model.body_gravcomp[ball_id] = 0.0 if gravity_on(data.time) else 1.0
    mujoco.mj_step(model, data)


def report(model, data):
    contacts = object_contacts(model, data, "ball")
    print(f"t={data.time:.2f}s  ball z={data.body('ball').xpos[2]:.3f} m  contacts={len(contacts)}")
    forces, counts = per_body_normal_force(contacts), per_body_count(contacts)
    for body in sorted(forces):
        print(f"  {body:10s} {forces[body]:5.2f} N  over {counts[body]:2d} contact points")


def run_headless(model, data, duration, controller, ball_id):
    t0 = time.time()
    while data.time < duration:
        step(model, data, controller, ball_id)
    report(model, data)
    print(f"({duration:.1f} s simulated in {time.time() - t0:.1f} s wall time)")


def run_viewer(model, data, duration, controller, ball_id):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            v.cam.lookat[:] = data.body("ball").xpos
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.18, 60, -15
        while v.is_running():
            start = time.time()
            if data.time >= duration:
                mujoco.mj_resetData(model, data)
            step(model, data, controller, ball_id)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--rigid", action="store_true", help="use the original rigid fingertips")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--squeeze", type=float, default=0.008, help="pad travel past first contact [m]")
    args = ap.parse_args()
    model = build_scene(pinch=True, soft_pads=not args.rigid)
    data = mujoco.MjData(model)
    controller, ball_id = PinchController(model, squeeze=args.squeeze), model.body("ball").id
    (run_headless if args.headless else run_viewer)(model, data, args.duration, controller, ball_id)
