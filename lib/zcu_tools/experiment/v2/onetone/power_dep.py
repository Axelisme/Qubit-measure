from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import snr_checker, sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program.v2 import (
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.process import minus_background, rescale


@dataclass(frozen=True)
class PowerDepResult:
    gains: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: PowerDepCfg | None = None


def gaindep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rescale(minus_background(np.abs(signals), axis=-1), axis=-1)


class PowerDepModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    readout: PulseReadoutCfg


class PowerDepSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class PowerDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerDepModuleCfg
    sweep: PowerDepSweepCfg


class PowerDepExp(PersistableExperiment[PowerDepResult, PowerDepCfg]):
    # freqs stored as Hz on disk (MHz * 1e6); gains stored as-is (IDENTITY)
    AXES_SPEC = AxesSpec(
        axes=(
            Axis("freqs", "Frequency", "Hz", scale=MHZ_TO_HZ),
            Axis("gains", "Power", "a.u.", scale=IDENTITY),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=PowerDepResult,
        cfg_type=PowerDepCfg,
        tag="onetone/power_dep",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: PowerDepCfg,
        *,
        earlystop_snr: float | None = None,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> PowerDepResult:
        orig_cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain

        readout_cfg = modules.readout
        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": readout_cfg.pulse_cfg.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            cfg.sweep.freq,
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": readout_cfg.pulse_cfg.ch,
                "ro_ch": readout_cfg.ro_cfg.ro_ch,
            },
        )

        current_snr = 0.0

        def update_snr(snr: float) -> None:
            nonlocal current_snr
            current_snr = snr

        with LivePlot2DwithLine(
            "Power (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals_buffer = SignalBuffer(
                (len(gains), len(freqs)),
                on_update=lambda data: viewer.update(
                    gains,
                    freqs,
                    gaindep_signal2real(data),
                    title=f"snr = {current_snr:.1f}" if current_snr else None,
                ),
            )
            with Schedule(cfg, signals_buffer) as sched:
                for _, step in sched.scan("gain", gains.tolist()):
                    cfg = step.cfg
                    modules = cfg.modules

                    modules.readout.set_param("gain", step.value)
                    freq_sweep = cfg.sweep.freq
                    modules.readout.set_param("freq", sweep2param("freq", freq_sweep))

                    _ = (
                        step.prog_builder(soc, soccfg)
                        .add_reset("reset", modules.reset)
                        .add(PulseReadout("readout", modules.readout))
                        .declare_sweep("freq", freq_sweep)
                        .build_and_acquire(
                            stop_condition=snr_checker(
                                signals_buffer[step],
                                earlystop_snr,
                                gaindep_signal2real,
                                after_check=update_snr,
                            ),
                            **(acquire_kwargs or {}),
                        )
                    )

        return PowerDepResult(
            gains=gains,
            freqs=freqs,
            signals=signals_buffer.array,
            cfg_snapshot=orig_cfg,
        )

    @retrieve_result
    def analyze(
        self,
        result: PowerDepResult | None = None,
    ) -> None:
        raise NotImplementedError("Not implemented")
