"""Pinch each object with the soft pads and save a gallery image: for every object a close-up
render plus the index and thumb tactile images at the end of the hold.

    python scripts/tactile_gallery.py                 # writes tactile_gallery.png
    python scripts/tactile_gallery.py --objects cube plate --out out.png
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_sim.objects import OBJECTS, SQUEEZE  # noqa: E402
from hand_sim.pinch import PinchController, gravity_on  # noqa: E402
from hand_sim.pressure import PadPressure  # noqa: E402
from hand_sim.scene import build_scene  # noqa: E402
from hand_sim.tactile_viz import add_pad_heat, taxel_image  # noqa: E402

PMAX = 6000.0
W, H, TILE = 400, 300, 140


def simulate(kind, t_end=5.0):
    model = build_scene(pinch=kind, soft_pads=True)
    data = mujoco.MjData(model)
    ctrl, obj = PinchController(model, squeeze=SQUEEZE[kind]), model.body("object").id
    while data.time < t_end:
        data.ctrl[:] = ctrl.ctrl(data.time)
        model.body_gravcomp[obj] = 0.0 if gravity_on(data.time) else 1.0
        mujoco.mj_step(model, data)
    pressure = PadPressure(model)
    pressure.update(model, data)
    return model, data, pressure


def panel(kind):
    model, data, pressure = simulate(kind)
    model.geom("object").rgba[3] = 0.35                       # see-through, so the pad heat shows
    r = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.body("object").xpos
    cam.distance, cam.azimuth, cam.elevation = 0.12, 150, -20
    r.update_scene(data, cam)
    for pad in pressure.pads:
        add_pad_heat(r.scene, pressure.face_positions(model, data, pad), pressure.taxels(pad), PMAX)
    render = r.render()
    r.close()
    tiles = [taxel_image(pressure.taxels(p), PMAX, TILE) for p in ("if_flex", "th_flex")]
    col = np.full((H + TILE + 40, W, 3), 20, np.uint8)
    col[:H] = render
    x0 = (W - 2 * TILE - 20) // 2
    col[H + 30:H + 30 + TILE, x0:x0 + TILE] = tiles[0]
    col[H + 30:H + 30 + TILE, x0 + TILE + 20:x0 + 2 * TILE + 20] = tiles[1]
    img = Image.fromarray(col)
    d = ImageDraw.Draw(img)
    held = "held" if data.body("object").xpos[2] > 0.2 else "DROPPED"
    d.text((10, 10), f"{kind} ({held})", fill=(255, 255, 255))
    d.text((x0, H + 12), "index", fill=(200, 200, 200))
    d.text((x0 + TILE + 20, H + 12), "thumb", fill=(200, 200, 200))
    print(f"{kind}: {held}, peak index {pressure.taxels('if_flex').max()/1000:.1f} kPa, "
          f"thumb {pressure.taxels('th_flex').max()/1000:.1f} kPa", flush=True)
    return np.asarray(img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=list(OBJECTS), choices=OBJECTS)
    ap.add_argument("--out", default="tactile_gallery.png")
    args = ap.parse_args()
    Image.fromarray(np.concatenate([panel(k) for k in args.objects], axis=1)).save(args.out)
    print(f"saved {args.out}  (colour scale 0-{PMAX/1000:.0f} kPa)")
