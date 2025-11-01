import warnings
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
from qick.asm_v2 import QickParam

from zcu_tools.utils import deepupdate
from zcu_tools.utils.process import rotate2real


def wrap_with_flux_pulse(
    pulse: Dict[str, Any], flx_cfg: Dict[str, Any], margin: float = 0.0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Derive a flux pulse waveform and parameters from a given pulse, and delay the pulse t if need
    It will not overrid the existing value in flx_cfg.
    """
    pulse = deepcopy(pulse)
    derive_pulse = {
        "nqz": 1,
        "freq": 0.0,
        "phase": 0.0,
        "outsel": "input",
        "block_mode": False,
    }

    def check_delay(delay: str) -> None:
        if isinstance(pulse[delay], QickParam):
            raise ValueError(
                f"pulse {delay} cannot be a QickParam when using flux pulse wrap."
            )

    check_delay("pre_delay")
    check_delay("post_delay")

    # derive t
    pre_delay = pulse["pre_delay"]
    if pre_delay >= margin:
        flux_pre_delay = pre_delay - margin
    else:
        flux_pre_delay = 0.0
        pre_delay = margin
    pulse["pre_delay"] = pre_delay

    # derive length
    flux_length = pulse["waveform"]["length"] + 2 * margin

    # derive post_delay
    post_delay = pulse["post_delay"]
    if post_delay >= margin:
        flux_post_delay = post_delay - margin
    else:
        flux_post_delay = 0.0
        post_delay = margin
    pulse["post_delay"] = post_delay

    derive_pulse.update(
        pre_delay=flux_pre_delay,
        length=flux_length,
        post_delay=flux_post_delay,
    )

    deepupdate(derive_pulse, flx_cfg, behavior="force")

    if derive_pulse["block_mode"]:
        warnings.warn(
            "Wrapped flux pulse is in block mode, this will block the inner pulse to start"
        )

    return pulse, derive_pulse


def check_flux_pulse(flx_cfg: Dict[str, Any], name: str = "flux_pulse") -> None:
    if flx_cfg["style"] not in ["const", "flat_top"]:
        raise ValueError(
            f"{name} style {flx_cfg['style']} not supported in flux sweep."
        )


def check_gains(gains: float, name: str) -> np.ndarray:
    if np.any(gains > 1.0):
        warnings.warn(
            f"Some {name} gains are larger than 1.0, force clip to 1.0, which may cause distortion."
        )
        gains = np.clip(gains, 0.0, 1.0)
    return gains


