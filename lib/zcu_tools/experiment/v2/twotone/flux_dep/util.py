from typing import Any, Dict
from zcu_tools.program.v2 import check_no_post_delay
from zcu_tools.utils import deepupdate


def derive_flux_pulse_from_pulse(
    pulse: Dict[str, Any], flx_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Derive a flux pulse waveform and parameters from a given pulse.
    It will not overrid the existing value in flx_cfg.
    """
    derive_pulse = {
        "style": "const",
        "length": pulse["length"],
        "nqz": 1,
        "freq": 0.0,
        "post_delay": None,
        "t": pulse.get("t", 0.0),
    }

    deepupdate(derive_pulse, flx_cfg, behavior="force")

    return derive_pulse


def check_flux_pulse(
    flx_cfg: Dict[str, Any], name: str = "flux_pulse", check_delay: bool = True
) -> None:
    if flx_cfg["style"] not in ["const", "flat_top"]:
        raise ValueError(
            f"Flux pulse style {flx_cfg['style']} not supported in flux sweep."
        )

    if check_delay:
        check_no_post_delay(flx_cfg, name)
