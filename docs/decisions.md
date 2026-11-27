# Decisions

## Simulator: MuJoCo
- Runs on CPU, so development on a Mac with no GPU.
- Menagerie hand models (LEAP, Allegro, Shadow); built-in contact point and force rendering; flex soft bodies.
- Same MJCF can later run on Newton (MuJoCo Warp backend) on an NVIDIA GPU.

## Fingertips: FEM (not PBD/XPBD or projective dynamics)
- Pads are small (hundreds to a few thousand tets), so speed isn't the constraint; force fidelity is.
- PBD gives no physical forces; XPBD forces depend on convergence, which would show up as noise in pressure maps.
- Projective dynamics: changing contact sets break the prefactorisation; limited material models for near-incompressible elastomers.
- FEM supports a credible sim-to-real tactile story (comparison against vision-based tactile sensors).

## Language: hybrid
- Python: scene construction, scripted grasps, plotting, video.
- C++: physics MuJoCo doesn't provide, as an engine plugin loaded from Python.

## Grasps: scripted
- Effort goes into contact and pressure visualisation, not policy learning.
