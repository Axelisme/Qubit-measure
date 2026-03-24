from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typeguard import check_type
from typing_extensions import Any, Optional, TypeAlias, cast

from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.utils import format_sweep1D, sweep2array
from zcu_tools.experiment.v2.runner import Task, TaskCfg, run_task, TaskState
from zcu_tools.experiment.v2.utils import round_zcu_time
from zcu_tools.liveplot import LivePlotter1D
from zcu_tools.program import SweepCfg
from zcu_tools.program.v2 import Pulse, TwoToneCfg, TwoToneProgram, sweep2param
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

# (lens, signals)
LenRabiResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LenRabiCfg(TwoToneCfg, TaskCfg):
    sweep: dict[str, SweepCfg]


class LenRabiExp(AbsExperiment[LenRabiResult, LenRabiCfg]):
    def _run_for_flat(self, soc, soccfg, cfg: dict[str, Any]) -> LenRabiResult:
        _cfg = check_type(deepcopy(cfg), LenRabiCfg)
        modules = _cfg["modules"]

        assert modules["qub_pulse"]["waveform"]["style"] in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        lengths = sweep2array(_cfg["sweep"]["length"])  # predicted

        Pulse.set_param(
            modules["qub_pulse"],
            "length",
            sweep2param("length", _cfg["sweep"]["length"]),
        )

        with LivePlotter1D("Length (us)", "Signal") as viewer:

            def measure_fn(ctx: TaskState, update_hook):
                nonlocal lengths
                prog = TwoToneProgram(soccfg, ctx.cfg)

                # get actual lengths after program generation, in case there are some adjustments
                lengths = cast(
                    NDArray[np.float64],
                    prog.get_pulse_param("qub_pulse", "length", as_array=True),
                )

                return prog.acquire(soc, progress=False, callback=update_hook)

            signals = run_task(
                task=Task(measure_fn=measure_fn, result_shape=(len(lengths),)),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, rabi_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def _run_for_arb(self, soc, soccfg, cfg: dict[str, Any]) -> LenRabiResult:
        _cfg = check_type(deepcopy(cfg), LenRabiCfg)
        modules = _cfg["modules"]

        length_sweep = _cfg["sweep"].pop("length")

        lengths = sweep2array(length_sweep)  # predicted
        lengths = round_zcu_time(lengths, soccfg, gen_ch=modules["qub_pulse"]["ch"])
        lengths = np.unique(lengths)  # remove duplicates

        with LivePlotter1D("Length (us)", "Signal") as viewer:

            def measure_fn(ctx: TaskState, update_hook):
                nonlocal lengths
                prog = TwoToneProgram(soccfg, ctx.cfg)

                # get actual lengths after program generation, in case there are some adjustments
                true_t = float(prog.get_time_param("pi2_pulse1", "t"))
                lengths[ctx.env["scan_index"]] = true_t

                return prog.acquire(soc, progress=False, callback=update_hook)

            def update_fn(i, ctx: TaskState, length):
                ctx.env["scan_index"] = i
                Pulse.set_param(ctx.cfg["modules"]["qub_pulse"], "length", length)

            signals = run_task(
                task=Task(measure_fn=measure_fn).scan(
                    "length",
                    lengths.tolist(),
                    before_each=lambda _, ctx, length: Pulse.set_param(
                        ctx.cfg["modules"]["qub_pulse"], "length", length
                    ),
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, rabi_signal2real(np.asarray(ctx.root_data))
                ),
            )
            signals = np.asarray(signals)

        # record last cfg and result
        self.last_cfg = _cfg
        self.last_result = (lengths, signals)

        return lengths, signals

    def run(self, soc, soccfg, cfg: dict[str, Any]) -> LenRabiResult:
        modules = cfg["modules"]
        qub_waveform = modules["qub_pulse"]["waveform"]

        cfg["sweep"] = format_sweep1D(cfg["sweep"], "length")

        if qub_waveform["style"] in ["const", "flat_top"]:
            # use hard sweep for flat top pulse
            return self._run_for_flat(soc, soccfg, cfg)
        else:
            # use soft sweep for arb pulse
            return self._run_for_arb(soc, soccfg, cfg)

    def analyze(
        self, result: Optional[LenRabiResult] = None, *, decay: bool = True
    ) -> tuple[float, float, float, Figure]:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result

        real_signals = rabi_signal2real(signals)

        nan_mask = np.isnan(real_signals)
        if np.all(nan_mask):
            raise ValueError("All data are NaN!")

        lens = lens[~nan_mask]
        real_signals = real_signals[~nan_mask]

        pi_len, pi2_len, freq, y_fit, _ = fit_rabi(
            lens, real_signals, decay=decay, init_phase=None
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lens, y_fit, label="fit")
        ax.axvline(pi_len, ls="--", c="red", label=f"pi = {pi_len:.3g} μs")
        ax.axvline(pi2_len, ls="--", c="red", label=f"pi/2 = {pi2_len:.3g} μs")
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(f"Rabi Oscillation (f={freq:.3f} MHz)")
        ax.legend(loc=4)
        ax.grid(True)

        fig.tight_layout()

        return pi_len, pi2_len, freq, fig

    def save(
        self,
        filepath: str,
        result: Optional[LenRabiResult] = None,
        comment: Optional[str] = None,
        tag: str = "twotone/ge/rabi_length",
        **kwargs,
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        lens, signals = result
        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LenRabiResult:
        signals, lens, _ = load_data(filepath, **kwargs)
        assert lens is not None
        assert len(lens.shape) == 1 and len(signals.shape) == 1
        assert lens.shape == signals.shape

        lens = lens * 1e6  # s -> us

        lens = lens.astype(np.float64)
        signals = signals.astype(np.complex128)

        self.last_cfg = None
        self.last_result = (lens, signals)

        return lens, signals
