from copy import deepcopy

from zcu_tools.program.v1 import RFreqTwoToneProgram, RFreqTwoToneProgramWithRedReset
from zcu_tools.schedule.tools import sweep2array
from zcu_tools.schedule.v1.template import sweep1D_hard_template


def measure_qub_freq(soc, soccfg, cfg, reset_rf=None):
    cfg = deepcopy(cfg)  # prevent in-place modification

    if reset_rf is not None:
        assert cfg.get("reset") == "pulse", "Need reset=pulse for conjugate reset"
        assert "reset_pulse" in cfg["dac"], "Need reset_pulse for conjugate reset"
        cfg["r_f"] = reset_rf

    fpts = sweep2array(cfg["sweep"])

    fpts, signals = sweep1D_hard_template(
        soc,
        soccfg,
        cfg,
        RFreqTwoToneProgram if reset_rf is None else RFreqTwoToneProgramWithRedReset,
        xs=fpts,
        xlabel="Frequency (MHz)",
        ylabel="Amplitude",
    )

    return fpts, signals
