# FEM fingertip plugin (planned)

A MuJoCo engine plugin in C++ that simulates each fingertip pad as a hyperelastic FEM mesh attached to the distal phalanx, and exposes the contact pressure field over the pad surface.

Planned shape:
- `CMakeLists.txt` building a shared library against the MuJoCo headers
- tetrahedral pad mesh per fingertip, Neo-Hookean (or corotated) material, implicit integration
- two-way coupling with the rigid finger link
- per-surface-vertex pressure written to plugin state / sensor data, read from Python for plotting

We'll first prototype soft pads with MuJoCo's built-in flex and only move to this plugin if the pressure resolution is too coarse.
