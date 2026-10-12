"""Workflow flux tools composed from live cfg reads and complete device setup."""

from collections.abc import Generator

from zcu_tools.experiment.utils import check_flux_unit
from zcu_tools.experiment.workflows import Effect, WorkflowEnv


def set_flux[C](
    env: WorkflowEnv[C], name: str, value: float, *, unit: str
) -> Generator[Effect, None, None]:
    """Set absolute flux in the explicitly declared stored coordinate.

    Live-read complete settings, check unit with the shared experiment owner,
    then copy only value and yield full setup. No unit conversion or mode change
    occurs. Mismatch and unsupported Info classes fail before setup; driver
    validation, limits and I/O causes propagate. Keep output and rampstep intact.
    """
    cfg = env.get_device_cfg(name)
    check_flux_unit(cfg, unit)
    yield from env.setup_device(name, cfg.model_copy(update={"value": value}))
