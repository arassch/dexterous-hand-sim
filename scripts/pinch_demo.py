"""Thumb-index pinch with soft FEM fingertip pads (sphere, cube, cylinder or plate).

The hand moves to a pre-grasp, then the index and thumb close straight towards each
other along their pad normals. The sphere floats until t=3.5 s, when gravity is
switched on and the pads must hold it by friction. Contact points and forces are shown in the viewer.

With soft pads, the viewer also shows contact pressure: coloured dots on each pad's
outer-face nodes and live 5x5 tactile images for the index (left) and thumb (right).

Interactive (macOS needs mjpython):
    mjpython scripts/pinch_demo.py
    mjpython scripts/pinch_demo.py --object cube   # or cylinder, plate, bar
    mjpython scripts/pinch_demo.py --object bar --rigid   # rigid tips drop the off-centre bar
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
from hand_sim.pressure import PadPressure  # noqa: E402
from hand_sim.tactile_viz import add_pad_heat, taxel_image  # noqa: E402

PMAX = 6000.0            # colour-scale max [Pa]
TACTILE_PADS = ("if_flex", "th_flex")
TACTILE_PX = 160
from hand_sim.objects import OBJECTS, SQUEEZE  # noqa: E402
from hand_sim.scene import build_scene  # noqa: E402


def step(model, data, controller, obj_id):
    data.ctrl[:] = controller.ctrl(data.time)
    model.body_gravcomp[obj_id] = 0.0 if gravity_on(data.time) else 1.0
    mujoco.mj_step(model, data)


def report(model, data):
    contacts = object_contacts(model, data, "object")
    print(f"t={data.time:.2f}s  object z={data.body('object').xpos[2]:.3f} m  contacts={len(contacts)}")
    forces, counts = per_body_normal_force(contacts), per_body_count(contacts)
    for body in sorted(forces):
        print(f"  {body:10s} {forces[body]:5.2f} N  over {counts[body]:2d} contact points")


def run_headless(model, data, duration, controller, obj_id, pressure=None, save=None):
    t0 = time.time()
    while data.time < duration:
        step(model, data, controller, obj_id)
    report(model, data)
    print(f"({duration:.1f} s simulated in {time.time() - t0:.1f} s wall time)")
    if pressure is not None:
        pressure.update(model, data)
        np.set_printoptions(precision=1, suppress=True, linewidth=120)
        for pad in TACTILE_PADS:
            print(f"{pad} taxels [kPa] (rows: finger base -> tip, cols: across finger)")
            print(pressure.taxels(pad) / 1000)
        if save:
            from PIL import Image
            imgs = [taxel_image(pressure.taxels(p), PMAX, TACTILE_PX) for p in TACTILE_PADS]
            Image.fromarray(np.concatenate(imgs, axis=1)).save(save)
            print(f"tactile images saved to {save}")


def run_viewer(model, data, duration, controller, obj_id, pressure=None):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v:
        with v.lock():
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            v.cam.lookat[:] = data.body("object").xpos
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 0.18, 60, -15
        while v.is_running():
            start = time.time()
            if data.time >= duration:
                mujoco.mj_resetData(model, data)
            step(model, data, controller, obj_id)
            if pressure is not None and int(round(data.time / model.opt.timestep)) % 10 == 0:
                draw_pressure(v, model, data, pressure)
            v.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))


def draw_pressure(v, model, data, pressure):
    pressure.update(model, data)
    with v.lock():
        v.user_scn.ngeom = 0
        for pad in pressure.pads:
            add_pad_heat(v.user_scn, pressure.face_positions(model, data, pad), pressure.taxels(pad), PMAX)
    vp = v.viewport
    if vp is None:
        return
    imgs = []
    for k, pad in enumerate(TACTILE_PADS):
        x0 = vp.left + vp.width - 2 * (TACTILE_PX + 10)          # bottom-right corner
        rect = mujoco.MjrRect(x0 + k * (TACTILE_PX + 10), vp.bottom + 10, TACTILE_PX, TACTILE_PX)
        imgs.append((rect, taxel_image(pressure.taxels(pad), PMAX, TACTILE_PX)))
    v.set_images(imgs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--rigid", action="store_true", help="use the original rigid fingertips")
    ap.add_argument("--object", default="sphere", choices=OBJECTS)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--squeeze", type=float, help="pad travel past first contact [m] (default per object)")
    ap.add_argument("--save-tactile", metavar="PNG", help="headless: save final tactile images")
    args = ap.parse_args()
    model = build_scene(pinch=args.object, soft_pads=not args.rigid)
    data = mujoco.MjData(model)
    squeeze = SQUEEZE[args.object] if args.squeeze is None else args.squeeze
    controller, obj_id = PinchController(model, squeeze=squeeze), model.body("object").id
    pressure = None if args.rigid else PadPressure(model)
    if args.headless:
        run_headless(model, data, args.duration, controller, obj_id, pressure, args.save_tactile)
    else:
        run_viewer(model, data, args.duration, controller, obj_id, pressure)
