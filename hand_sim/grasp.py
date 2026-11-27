"""Scripted grasp: settle the object on the palm, then ramp the fingers closed."""
import numpy as np


def grasp_target(model, finger: float = 0.9, thumb: float = 0.8) -> np.ndarray:
    tgt = np.zeros(model.nu)
    for i in range(model.nu):
        name = model.actuator(i).name
        if "rot" in name:
            tgt[i] = 0.0                      # keep finger abduction neutral
        else:
            tgt[i] = thumb if name.startswith("th_") else finger
    return np.clip(tgt, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def ctrl_at(t: float, target: np.ndarray, settle: float = 1.5, close: float = 4.0) -> np.ndarray:
    """Open hand until `settle` seconds, then close linearly over `close` seconds."""
    a = np.clip((t - settle) / close, 0.0, 1.0)
    return a * target
