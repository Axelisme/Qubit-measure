from typing import Any, Dict, Tuple
from copy import deepcopy

from zcu_tools.program.v2 import check_no_post_delay
from zcu_tools.utils import deepupdate

from qick.asm_v2 import QickParam


def wrap_with_flux_pulse(
    pulse: Dict[str, Any], flx_cfg: Dict[str, Any], margin: float = 0.0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Derive a flux pulse waveform and parameters from a given pulse, and delay the pulse t if need
    It will not overrid the existing value in flx_cfg.
    """
    pulse = deepcopy(pulse)
    derive_pulse = {"nqz": 1, "freq": 0.0, "phase": 0.0}

    if isinstance(pulse["t"], QickParam):
        raise ValueError("pulse t cannot be a QickParam when using flux pulse wrap.")
    if isinstance(pulse["post_delay"], QickParam):
        raise ValueError(
            "pulse post_delay cannot be a QickParam when using flux pulse wrap."
        )

    # derive t
    pulse_t = pulse["t"]
    if pulse_t >= margin:
        flux_t = pulse_t - margin
    else:
        flux_t = 0.0
        pulse_t = margin
    pulse["t"] = pulse_t

    # derive length
    flux_length = pulse["length"] + 2 * margin

    # derive post_delay
    pulse_post_delay = pulse["post_delay"]
    if pulse_post_delay is not None and pulse_post_delay >= margin:
        pulse_post_delay = pulse_post_delay - margin

    derive_pulse.update(t=flux_t, length=flux_length, post_delay=None)

    deepupdate(derive_pulse, flx_cfg, behavior="force")

    return pulse, derive_pulse


def check_flux_pulse(
    flx_cfg: Dict[str, Any], name: str = "flux_pulse", check_delay: bool = True
) -> None:
    if flx_cfg["style"] not in ["const", "flat_top"]:
        raise ValueError(
            f"Flux pulse style {flx_cfg['style']} not supported in flux sweep."
        )

    if check_delay:
        check_no_post_delay(flx_cfg, name)
