from __future__ import annotations

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing_extensions import Any, Callable, Optional, TypeAlias

from zcu_tools.config import ConfigBase
from zcu_tools.experiment import AbsExperiment, config
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.v2 import (
    Pulse,
    SweepCfg,
    TwoToneCfg,
    TwoToneProgram,
    sweep2param,
)
from zcu_tools.utils.datasaver import load_data, save_data
from zcu_tools.utils.fitting import fit_rabi
from zcu_tools.utils.process import rotate2real

# (lens, signals)
LenRabiResult: TypeAlias = tuple[NDArray[np.float64], NDArray[np.complex128]]


def rabi_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    return rotate2real(signals).real


class LenRabiSweepCfg(ConfigBase):
    length: SweepCfg


class LenRabiCfg(TwoToneCfg, ExpCfgModel):
    sweep: LenRabiSweepCfg


class LenRabiExp(AbsExperiment[LenRabiResult, LenRabiCfg]):
    def _run_for_flat(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> LenRabiResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        assert modules.qub_pulse.waveform.style in ["const", "flat_top"], (
            "This method only supports const and flat_top pulse style"
        )

        # initial values, may be rounded later
        lengths = sweep2array(
            cfg.sweep.length,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, LenRabiCfg],
            update_hook: Optional[Callable],
        ) -> list[NDArray[np.float64]]:
            cfg = ctx.cfg
            modules = cfg.modules

            length_sweep = cfg.sweep.length
            length_param = sweep2param("length", length_sweep)
            modules.qub_pulse.set_param("length", length_param)

            return TwoToneProgram(
                soccfg, cfg, sweep=[("length", length_sweep)]
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                **(acquire_kwargs or {}),
            )

        with LivePlot1D("Length (us)", "Signal") as viewer:
            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(lengths),),
                    pbar_n=cfg.rounds,
                ),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, rabi_signal2real(ctx.root_data)
                ),
            )

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, signals)

        return lengths, signals

    def _run_for_arb(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> LenRabiResult:
        setup_devices(cfg, progress=True)
        modules = cfg.modules

        rounds = cfg.rounds
        _cfg = cfg.model_copy(deep=True)
        _cfg.rounds = 1  # we'll handle the rounds in the task loop

        length_sweep = _cfg.sweep.length

        lengths = sweep2array(
            length_sweep,
            "time",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )
        lengths = np.unique(lengths)  # remove duplicates

        prog_cache: dict[float, TwoToneProgram] = {}

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, LenRabiCfg], update_hook
        ):
            cfg = ctx.cfg
            modules = cfg.modules

            length = float(modules.qub_pulse.waveform.length)

            if length not in prog_cache:
                prog_cache[length] = TwoToneProgram(soccfg, cfg)

            return prog_cache[length].acquire(
                soc, progress=False, round_hook=update_hook, **(acquire_kwargs or {})
            )

        def average_round(
            signals: list[list[NDArray[np.complex128]]],
        ) -> NDArray[np.complex128]:
            _signals = np.asarray(signals)  # shape: (rounds, len(lengths))
            mask = np.any(~np.isnan(_signals), axis=0)
            mean_signals = np.full(_signals.shape[1], np.nan, dtype=np.complex128)
            mean_signals[mask] = np.nanmean(_signals[:, mask], axis=0)
            return mean_signals

        with LivePlot1D("Length (us)", "Signal") as viewer:
            signals = run_task(
                task=Task(measure_fn=measure_fn, pbar_n=1)
                .scan(
                    "length",
                    lengths.tolist(),
                    before_each=lambda _, ctx, length: (
                        ctx.cfg.modules.qub_pulse.set_param("length", length)
                    ),
                )
                .scan(
                    "round",  # implement round loop here
                    list(range(rounds)),
                    before_each=lambda *_: None,
                ),
                init_cfg=_cfg,
                on_update=lambda ctx: viewer.update(
                    lengths, rabi_signal2real(average_round(ctx.root_data))
                ),
            )
            signals = average_round(signals)

        # record last cfg and result
        self.last_cfg = deepcopy(cfg)
        self.last_result = (lengths, signals)

        return lengths, signals

    def run(
        self,
        soc,
        soccfg,
        cfg: LenRabiCfg,
        *,
        acquire_kwargs: Optional[dict[str, Any]] = None,
    ) -> LenRabiResult:
        modules = cfg.modules
        qub_waveform = modules.qub_pulse.waveform

        if qub_waveform.style in ["const", "flat_top"]:
            # use hard sweep for flat top pulse
            return self._run_for_flat(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)
        else:
            # use soft sweep for arb pulse
            return self._run_for_arb(soc, soccfg, cfg, acquire_kwargs=acquire_kwargs)

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

        pi_len, pi_len_err, pi2_len, pi2_len_err, freq, freq_err, y_fit, _ = fit_rabi(
            lens, real_signals, decay=decay, init_phase=None
        )

        fig, ax = plt.subplots(figsize=config.figsize)
        assert isinstance(fig, Figure)

        ax.plot(lens, real_signals, label="meas", ls="-", marker="o", markersize=3)
        ax.plot(lens, y_fit, label="fit")
        ax.axvline(
            pi_len,
            ls="--",
            c="red",
            label=f"pi = {pi_len:.3g} ± {pi_len_err:.2g} μs",
        )
        ax.axvspan(pi_len - pi_len_err, pi_len + pi_len_err, color="red", alpha=0.2)
        ax.axvline(
            pi2_len,
            ls="--",
            c="red",
            label=f"pi/2 = {pi2_len:.3g} ± {pi2_len_err:.2g} μs",
        )
        ax.axvspan(pi2_len - pi2_len_err, pi2_len + pi2_len_err, color="red", alpha=0.2)
        ax.set_xlabel("Pulse length (μs)")
        ax.set_ylabel("Signal Real (a.u.)")
        ax.set_title(f"Rabi Oscillation (f={freq:.3f} ± {freq_err:.3f} MHz)")
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
        cfg = self.last_cfg
        assert cfg is not None
        comment = make_comment(cfg, comment)

        save_data(
            filepath=filepath,
            x_info={"name": "Length", "unit": "s", "values": lens * 1e-6},
            z_info={"name": "Signal", "unit": "a.u.", "values": signals},
            comment=comment,
            tag=tag,
            **kwargs,
        )

    def load(self, filepath: str, **kwargs) -> LenRabiResult:
        signals, lens, _, comment = load_data(filepath, return_comment=True, **kwargs)
        assert lens is not None
        assert len(lens.shape) == 1 and len(signals.shape) == 1
        assert lens.shape == signals.shape

        lens = lens * 1e6  # s -> us

        lens = lens.astype(np.float64)
        signals = signals.astype(np.complex128)

        if comment is not None:

            cfg, _, _ = parse_comment(comment)

            if cfg is not None:

                self.last_cfg = LenRabiCfg.validate_or_warn(cfg, source=filepath)
        self.last_result = (lens, signals)

        return lens, signals
