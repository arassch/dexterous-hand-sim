# dexterous-hand-sim

Simulation of a dexterous robot hand (LEAP hand) grasping objects, with **deformable FEM fingertips** and visualisation of **contact points, contact forces and fingertip pressure** (tactile-sensor style).

Built on [MuJoCo](https://github.com/google-deepmind/mujoco): Python for scene setup, scripted grasps and plotting; C++ for the physics that MuJoCo doesn't provide (a custom FEM fingertip plugin).

## Status

- [x] LEAP hand from MuJoCo Menagerie, scripted grasp of a ball
- [x] Contact points / force arrows in the viewer, per-finger normal forces headless
- [ ] Soft fingertips with MuJoCo flex (built-in elasticity)
- [ ] Fingertip pressure maps
- [ ] Custom C++ FEM fingertip plugin (`plugin/`)
- [ ] Video + write-up

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/fetch_models.sh          # sparse-clones the LEAP hand from MuJoCo Menagerie

mjpython scripts/grasp_demo.py     # interactive viewer (macOS needs mjpython)
python scripts/grasp_demo.py --headless
```

Headless output looks like:

```
t=8.00s  ball at [-0.054  0.019  0.179]  contacts=3
  mf_ds      1.81 N
  palm       8.68 N
  th_ds     10.35 N
```

## Layout

```
hand_sim/      Python package: scene building (MjSpec), contact extraction, grasp scripts
scripts/       entry points and model download
plugin/        C++ MuJoCo engine plugin for FEM fingertips (planned)
docs/          design notes and decisions
```

See [docs/decisions.md](docs/decisions.md) for why MuJoCo + FEM.
