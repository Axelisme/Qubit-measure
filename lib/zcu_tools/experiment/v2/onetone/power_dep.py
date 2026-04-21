from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import (
    Any,
    Callable,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
    cast,
)

from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskCfg, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array, wrap_earlystop_check
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import (
    ModularProgramCfg,
    ModularProgramV2,
    PulseReadout,
    PulseReadoutCfg,
    Reset,
    ResetCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.process import minus_background, rescale

PowerDepResult: TypeAlias = tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]
]


def gaindep_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rescale(minus_background(np.abs(signals), axis=1), axis=1)


class PowerDepModuleCfg(TypedDict, closed=True):
    reset: NotRequired[ResetCfg]
    readout: PulseReadoutCfg


class PowerDepCfg(ModularProgramCfg, TaskCfg):
    modules: PowerDepModuleCfg
    sweep: dict[str, SweepCfg]


class PowerDepExp(AbsExperiment[PowerDepResult, PowerDepCfg]):
    def run(
        self, soc, soccfg, cfg: dict[str, Any], *, earlystop_snr: Optional[float] = None
    ) -> PowerDepResult:
        _cfg = check_type(deepcopy(cfg), PowerDepCfg)
        setup_devices(_cfg, progress=True)
        modules = _cfg["modules"]

        gain_sweep = _cfg["sweep"]["gain"]

        readout_cfg = modules["readout"]
        gains = sweep2array(
            gain_sweep,
            "gain",
            {"soccfg": soccfg, "gen_ch": readout_cfg.pulse_cfg.ch},
            allow_array=True,
        )
        freqs = sweep2array(
            _cfg["sweep"]["freq"],
            "freq",
            {
                "soccfg": soccfg,
                "gen_ch": readout_cfg.pulse_cfg.ch,
                "ro_ch": readout_cfg.ro_cfg.ro_ch,
            },
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any],
            update_hook: Optional[Callable[[int, list[NDArray[np.float64]]], None]],
        ) -> list[NDArray[np.float64]]:
            cfg: PowerDepCfg = cast(PowerDepCfg, ctx.cfg)
            modules = cfg["modules"]

            assert update_hook is not None

            freq_sweep = cfg["sweep"]["freq"]
            freq_param = sweep2param("freq", freq_sweep)
            modules["readout"].set_param("freq", freq_param)

            return (
                prog := ModularProgramV2(
                    soccfg,
                    cfg,
                    modules=[
                        Reset("reset", modules.get("reset")),
                        PulseReadout("readout", modules["readout"]),
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
                    signal2real_fn=np.abs,
                    after_check=lambda snr: ax1d.set_title(f"snr = {snr:.1f}"),
                ),
            )

        # run experiment
        with LivePlot2DwithLine(
            "Power (a.u.)", "Frequency (MHz)", line_axis=1, num_lines=10
        ) as viewer:
            ax1d = viewer.get_ax("1d")

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(freqs),),
                    pbar_n=_cfg["rounds"],
                ).scan(
                    "gain",
                    gains.tolist(),
                    before_each=lambda _, ctx, gain: ctx.cfg["modules"][
                        "readout"
                    ].set_param("gain", gain),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    gains, freqs, gaindep_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
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
        signals2D, freqs, gains = load_data(filepath, **kwargs)
        assert freqs is not None and gains is not None
        assert len(freqs.shape) == 1 and len(gains.shape) == 1
        assert signals2D.shape == (len(gains), len(freqs))

        freqs = freqs * 1e-6  # Hz -> MHz

        gains = gains.astype(np.float64)
        freqs = freqs.astype(np.float64)
        signals2D = signals2D.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (gains, freqs, signals2D)

        return gains, freqs, signals2D
