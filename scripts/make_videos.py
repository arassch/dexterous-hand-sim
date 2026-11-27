"""Render demo videos (MP4) offscreen.

    python scripts/make_videos.py                      # all clips into videos/
    python scripts/make_videos.py --clips pinch_cube rigid_vs_soft --out videos

Clips:
  power_grasp      rigid-tip power grasp of a ball, contact points and forces
  pinch_<object>   soft-pad pinch of sphere/cube/cylinder/plate with pad pressure and tactile images
  rigid_vs_soft    sphere pinch, rigid tips (left) vs soft FEM pads (right)
  bar_rigid_vs_soft  bar held off-centre: rigid tips drop it, soft pads hold it

Rendering is offline, so the soft-pad clips play at real speed even though the
simulation runs slower than real time. Needs imageio[ffmpeg] (in requirements.txt).
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_sim.contacts import object_contacts  # noqa: E402
from hand_sim.grasp import ctrl_at, grasp_target  # noqa: E402
from hand_sim.objects import OBJECTS, SQUEEZE  # noqa: E402
from hand_sim.pinch import PINCH_POINT, T_RELEASE, PinchController, gravity_on  # noqa: E402
from hand_sim.pressure import PadPressure  # noqa: E402
from hand_sim.scene import build_scene  # noqa: E402
from hand_sim.tactile_viz import add_pad_heat, taxel_image  # noqa: E402

FPS = 30
W, H = 1280, 720
PMAX = 6000.0
TILE = 200


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1
        return ImageFont.load_default()


class Sim:
    """One scene + controller + renderer, stepped frame by frame."""

    def __init__(self, pinch, soft, width=W, height=H):
        self.model = build_scene(pinch=pinch, soft_pads=soft)
        m = self.model
        m.vis.global_.offwidth, m.vis.global_.offheight = max(width, 640), max(height, 480)
        self.data = mujoco.MjData(m)
        self.pinch, self.soft = pinch, soft
        self.obj = m.body("object").id
        self.ctrl = PinchController(m, squeeze=SQUEEZE[pinch]) if pinch else None
        self.target = None if pinch else grasp_target(m)
        self.pressure = PadPressure(m) if soft else None
        self.renderer = mujoco.Renderer(m, height, width)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.alpha0 = m.geom("object").rgba[3]

    def advance(self, t):
        m, d = self.model, self.data
        while d.time < t:
            if self.pinch:
                d.ctrl[:] = self.ctrl.ctrl(d.time)
                m.body_gravcomp[self.obj] = 0.0 if gravity_on(d.time) else 1.0
            else:
                d.ctrl[:] = ctrl_at(d.time, self.target)
            mujoco.mj_step(m, d)

    def render(self, azimuth, distance, elevation=-18, lookat=None, xray=0.0, contacts=False):
        m, d = self.model, self.data
        m.geom("object").rgba[3] = self.alpha0 * (1 - xray) + 0.3 * xray
        self.cam.lookat[:] = d.body("object").xpos if lookat is None else lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contacts
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contacts
        self.renderer.update_scene(d, self.cam, self.opt)
        if self.pressure is not None:
            self.pressure.update(m, d)
            for pad in self.pressure.pads:
                add_pad_heat(self.renderer.scene, self.pressure.face_positions(m, d, pad),
                             self.pressure.taxels(pad), PMAX, radius=0.0015)
        return self.renderer.render().copy()

    def tiles(self, size=TILE):
        return [taxel_image(self.pressure.taxels(p), PMAX, size) for p in ("if_flex", "th_flex")]


def _label(img, lines, xy=(24, 20), size=30, color=(255, 255, 255)):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = _font(size)
    y = xy[1]
    for line in lines:
        d.text((xy[0], y), line, font=f, fill=color)
        y += int(size * 1.3)
    return np.array(im)


def _paste_tiles(img, tiles, labels=("index", "thumb")):
    h, w = img.shape[:2]
    pad = 16
    for k, (tile, lab) in enumerate(zip(tiles, labels)):
        x0 = w - (len(tiles) - k) * (tile.shape[1] + pad)
        y0 = h - tile.shape[0] - pad
        img[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile
        img = _label(img, [lab], (x0 + 6, y0 - 34), size=24, color=(220, 220, 220))
    ts = tiles[0].shape[0]
    img = _label(img, [f"pad pressure 0-{PMAX / 1000:.0f} kPa"],
                 (w - len(tiles) * (ts + pad), h - ts - pad - 66), size=20, color=(180, 180, 180))
    return img


def _status(t):
    return "gravity OFF" if t < T_RELEASE else "gravity ON"


def clip_pinch(kind, duration=6.5):
    sim = Sim(kind, soft=True)
    frames = []
    for i in range(int(duration * FPS)):
        t = i / FPS
        sim.advance(t)
        az = 50 + 80 * t / duration                               # orbit past both pads
        xray = np.clip((t - 2.6) / 0.8, 0, 1)                     # fade object once pads touch
        img = sim.render(az, 0.17, elevation=-15, xray=xray)
        img = _paste_tiles(img, sim.tiles())
        img = _label(img, [f"Soft FEM fingertips - {kind}", f"t = {t:4.1f} s   {_status(t)}"])
        frames.append(img)
    return frames


def clip_power_grasp(duration=8.0):
    sim = Sim(None, soft=False)
    frames = []
    for i in range(int(duration * FPS)):
        t = i / FPS
        sim.advance(t)
        img = sim.render(110 + 50 * t / duration, 0.36, -25, lookat=[0.0, -0.03, 0.18], contacts=True)
        frames.append(_label(img, ["LEAP hand - power grasp (rigid tips)", f"t = {t:4.1f} s"]))
    return frames


def clip_rigid_vs_soft(kind="sphere", duration=6.5):
    half = (W // 2, H)
    rigid, soft = Sim(kind, soft=False, width=half[0]), Sim(kind, soft=True, width=half[0])
    frames = []
    for i in range(int(duration * FPS)):
        t = i / FPS
        rigid.advance(t)
        soft.advance(t)
        az = 60 + 40 * t / duration
        xray = np.clip((t - 2.6) / 0.8, 0, 1)
        a = rigid.render(az, 0.13, xray=xray, contacts=True)
        b = soft.render(az, 0.13, xray=xray)
        nr = len(object_contacts(rigid.model, rigid.data, "object"))
        ns = len(object_contacts(soft.model, soft.data, "object"))
        a = _label(a, ["Rigid fingertips", f"{nr} contact point{'s' * (nr != 1)}"])
        b = _label(b, ["Soft FEM fingertips", f"{ns} contact point{'s' * (ns != 1)}"])
        b = _paste_tiles(b, soft.tiles(150))
        img = np.concatenate([a, b], axis=1)
        img[:, W // 2 - 2:W // 2 + 2] = 255
        frames.append(_label(img, [f"t = {t:4.1f} s", _status(t)], (24, H - 90), size=26))
    return frames


def _bar_tilt(sim):
    a = sim.data.body("object").xmat.reshape(3, 3)[:, 1]          # bar long axis
    return float(np.degrees(np.arcsin(np.clip(-a[2], -1, 1))))


def clip_bar_rigid_vs_soft(duration=8.0):
    """Bar pinched 3 cm off-centre: gravity twists it about the pinch axis. Two rigid point
    contacts can't resist that torque and the bar swings and drops; the soft pads' contact
    patches resist it through distributed friction."""
    half = W // 2
    rigid, soft = Sim("bar", soft=False, width=half), Sim("bar", soft=True, width=half)
    look = np.array(PINCH_POINT) + [0.0, 0.02, -0.01]
    frames = []
    for i in range(int(duration * FPS)):
        t = i / FPS
        rigid.advance(t)
        soft.advance(t)
        panels = []
        for sim, title in ((rigid, "Rigid fingertips"), (soft, "Soft FEM fingertips")):
            img = sim.render(180, 0.30, -20, lookat=look, contacts=sim is rigid)
            dropped = sim.data.body("object").xpos[2] < 0.15
            state = "DROPPED" if dropped else f"bar tilt {_bar_tilt(sim):5.1f} deg"
            img = _label(img, [title, state], color=(255, 120, 120) if dropped else (255, 255, 255))
            if sim is soft:
                img = _paste_tiles(img, soft.tiles(150))
            panels.append(img)
        img = np.concatenate(panels, axis=1)
        img[:, half - 2:half + 2] = 255
        img = _label(img, ["bar held 3 cm off-centre", f"t = {t:4.1f} s", _status(t)], (24, H - 125), size=26)
        frames.append(img)
    return frames


CLIPS = {"power_grasp": clip_power_grasp, "rigid_vs_soft": clip_rigid_vs_soft,
         "bar_rigid_vs_soft": clip_bar_rigid_vs_soft,
         **{f"pinch_{k}": (lambda k=k: clip_pinch(k)) for k in OBJECTS}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=list(CLIPS), choices=list(CLIPS))
    ap.add_argument("--out", default="videos")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name in args.clips:
        frames = CLIPS[name]()
        path = out / f"{name}.mp4"
        imageio.mimwrite(path, frames, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                         pixelformat="yuv420p", ffmpeg_params=["-preset", "veryfast"])
        print(f"wrote {path} ({len(frames)} frames)", flush=True)
