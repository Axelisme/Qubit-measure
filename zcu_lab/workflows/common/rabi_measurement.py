"""Plan the calibration time grid on the active drive clock before acquisition."""

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.utils.time_sweep import plan_time_sweep

from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiResult
from zcu_lab.workflows.common.calibration_acquire import run_len_rabi


def measure_planned_rabi(
    cfg: LenRabiCfg, *, context: RunContext, retry: int
) -> LenRabiResult:
    """Plan cfg.length in us, then measure Length-Rabi on the active QICK clock.

    The workflow host supplies an attempt-owned cfg. Update its length before
    calling run_len_rabi so compilation and native RunRecord.cfg use the planned
    SweepCfg. The caller keeps the original requested sweep separately. retry is
    the workflow acquisition retry count. Planning failures propagate before
    acquisition; cancellation retains partial data as in run_len_rabi.
    """
    cfg.length = plan_time_sweep(cfg.length, context.soccfg, gen_ch=cfg.qub_pulse.ch)
    return run_len_rabi(cfg, context=context, retry=retry)
