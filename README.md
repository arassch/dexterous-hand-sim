# dexterous-hand-sim

Simulation of a dexterous robot hand (LEAP hand) grasping objects, with **deformable FEM fingertips** and visualisation of **contact points, contact forces and fingertip pressure** (tactile-sensor style).

Built on [MuJoCo](https://github.com/google-deepmind/mujoco): Python for scene setup, scripted grasps and plotting; C++ for the physics that MuJoCo doesn't provide (a custom FEM fingertip plugin).

## Status

- [x] LEAP hand from MuJoCo Menagerie, scripted grasp of a ball
- [x] Contact points / force arrows in the viewer, per-finger normal forces headless
- [x] Soft FEM fingertip pads with MuJoCo flex covering the whole palmar face of each tip and curling over the end (back half pinned, front half elastic)
- [x] Thumb–index pinch held by the pads alone (`scripts/pinch_demo.py`, `--rigid` to compare): pads directly opposed, each fingertip closes straight along its pad normal (Jacobian-planned)
- [ ] Fingertip pressure maps
- [ ] Custom C++ FEM fingertip plugin (`plugin/`)
- [ ] Video + write-up

## Quick start

Requires Python 3.10+ (macOS's built-in `python3` is 3.9, which has no MuJoCo wheels and fails trying to build from source).

```bash
pyenv install -s 3.12 && pyenv local 3.12   # or use Homebrew's python3.12
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/fetch_models.sh          # sparse-clones the LEAP hand from MuJoCo Menagerie

mjpython scripts/grasp_demo.py     # power grasp, rigid tips (macOS needs mjpython)
mjpython scripts/pinch_demo.py     # pinch with soft FEM pads (runs ~0.4x real time)
mjpython scripts/pinch_demo.py --rigid
python scripts/pinch_demo.py --headless
```

Pinch, headless — soft pads give contact *patches*, rigid tips give single points:

```
soft:   if_flex 0.93 N over 28 contact points | th_flex 0.90 N over 24 contact points
rigid:  if_ds   0.82 N over  1 contact point  | th_ds   0.80 N over  1 contact point
```

Pad stiffness, size and mesh resolution are parameters of `add_soft_pads()` in `hand_sim/soft_pads.py`
(resolution 5 ≈ 0.4x real time on CPU).

## Layout

```
hand_sim/      Python package: scene building (MjSpec), contact extraction, grasp scripts
scripts/       entry points and model download
plugin/        C++ MuJoCo engine plugin for FEM fingertips (planned)
docs/          design notes and decisions
```

See [docs/decisions.md](docs/decisions.md) for why MuJoCo + FEM.
