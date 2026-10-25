"""Plan the calibration time grid on the active drive clock before acquisition."""

from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.utils.time_sweep import plan_time_sweep

from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiCfg, LenRabiExp, LenRabiResult


def measure_planned_rabi(cfg: LenRabiCfg, *, context: RunContext) -> LenRabiResult:
    """Plan cfg.length in us, then measure Length-Rabi on the active QICK clock.

    The workflow host supplies an attempt-owned cfg. Update its length before
    calling the core so both compilation and native RunRecord.cfg use the planned
    SweepCfg. The caller keeps the original requested sweep separately. Planning
    failures propagate before acquisition; cancellation follows LenRabiExp.run.
    """
    cfg.length = plan_time_sweep(cfg.length, context.soccfg, gen_ch=cfg.qub_pulse.ch)
    return LenRabiExp().run(cfg, context=context)
