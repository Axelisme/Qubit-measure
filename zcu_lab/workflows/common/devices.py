"""Workflow flux tools composed from live cfg reads and complete device setup."""

from zcu_tools.experiment.scope import Scope

from zcu_lab.v2._support.device import check_flux_unit


async def set_flux(scope: Scope, name: str, value: float, *, unit: str) -> None:
    """Set absolute flux in the explicitly declared stored coordinate.

    Live-read complete settings, check unit with the shared experiment owner,
    then copy only value and await full setup. No unit conversion or mode change
    occurs. Mismatch and unsupported Info classes fail before setup; driver
    validation, limits and I/O causes propagate. Keep output and rampstep intact.
    """
    cfg = scope.get_device_cfg(name)
    check_flux_unit(cfg, unit)
    await scope.setup_device(name, cfg.model_copy(update={"value": value}))
