from typing import Any, Dict
from zcu_tools.program.v2 import check_no_post_delay


def check_flux_pulse(flx_cfg: Dict[str, Any], name: str = "flux_pulse") -> None:
    if flx_cfg["style"] not in ["const", "flat_top"]:
        raise ValueError(
            f"Flux pulse style {flx_cfg['style']} not supported in flux sweep."
        )

    check_no_post_delay(flx_cfg, name)
