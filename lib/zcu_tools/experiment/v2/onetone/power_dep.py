from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program.v2 import (
    ModularProgramV2,
    ProgramV2Cfg,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background, rescale

PowerDepResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def gaindep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rescale(minus_background(np.abs(signals), axis=-1), axis=-1)


class PowerDepModuleCfg(ConfigBase):
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg


class PowerDepSweepCfg(ConfigBase):
    gain: SweepCfg
    freq: SweepCfg


class PowerDepCfg(ProgramV2Cfg, ExpCfgModel):
    modules: PowerDepModuleCfg
    sweep: PowerDepSweepCfg


class PowerDepExp(AbsExperiment[PowerDepResult, PowerDepCfg]):
    def run(
        self, soc, soccfg, cfg: PowerDepCfg, *, earlystop_snr: Optional[float] = None
    ) -> PowerDepResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        gain_sweep = cfg.sweep.gain

        readout_cfg = modules.readout
        gains = sweep2array(
            gain_sweep,
            "gain",
            {
                "soccfg": soccfg,
                "gen_ch": readout_cfg.pulse_cfg.ch,
                "ro_ch": readout_cfg.ro_cfg.ro_ch,
            },
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

        def measure_fn(
            ctx: TaskState[Any, Any, PowerDepCfg], update_hook: Optional[Callable]
        ) -> list[NDArray[np.float64]]:
            nonlocal current_snr

            cfg = ctx.cfg
            modules = cfg.modules

            assert update_hook is not None

            freq_sweep = cfg.sweep.freq
            freq_param = sweep2param("freq", freq_sweep)
            modules.readout.set_param("freq", freq_param)

            def update_snr(snr: float) -> None:
                nonlocal current_snr
                current_snr = snr

            return (
                prog := ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", modules.reset),
                        PulseReadout("readout", modules.readout),
                    ],
                    sweep=[("freq", freq_sweep)],
                )
            ).acquire(
                soc,
                progress=False,
                round_hook=wrap_earlystop_check(
                    prog,
                    update_hook,
                    earlystop_snr,
                    signal2real_fn=gaindep_signal2real,
                    after_check=update_snr,
                ),
            )

        # run experiment
        with LivePlot2DwithLine(
            "Power (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=cfg.rounds,
                ).scan(
                    "gain",
                    gains.tolist(),
                    before_each=lambda _, ctx, gain: ctx.cfg.modules.readout.set_param(
                        "gain", gain
                    ),
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    gains,
                    freqs,
                    gaindep_signal2real(np.asarray(ctx.root_data)),
                    title=f"snr = {current_snr:.1f}" if current_snr else None,
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (gains, freqs, signals)

        return gains, freqs, signals

    def analyze(
        self,
        result: Optional[PowerDepResult] = None,
    ) -> None:
        raise NotImplementedError("Not implemented")

    def save(
        self,
        filepath: str,
        result: Optional[PowerDepResult] = None,
        comment: Optional[str] = None,
        tag: str = "onetone/power_dep",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        gains, freqs, signals2D = result

        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Frequency", "unit": "Hz", "values": freqs * 1e6},
            y_info={"name": "Power", "unit": "a.u.", "values": gains},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals2D},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> PowerDepResult:
        signals2D, freqs, gains, comment = load_data(
            filepath, return_comment=True, **kwargs
        )
        assert freqs is not None and gains is not None
        assert len(freqs.shape) == 1 and len(gains.shape) == 1
        assert signals2D.shape == (len(gains), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        if comment is not None:
            cfg, _, _ = parse_comment(comment)
            if cfg is not None:
                self.last_cfg = PowerDepCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (gains, freqs, signals2D)

        return gains, freqs, signals2D
