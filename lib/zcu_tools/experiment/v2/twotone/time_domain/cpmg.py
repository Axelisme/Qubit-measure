from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    US_TO_S,
    AbsExperiment,
    GroupedAxesSpec,
    GroupedLoadData,
    RoleAxisSpec,
    RoleSpec,
    RoleZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, setup_devices
from zcu_tools.experiment.v2.runner import Task, TaskState, run_task
from zcu_tools.experiment.v2.utils import snr_checker, sweep2array
from zcu_tools.liveplot import LivePlot2DwithLine
from zcu_tools.notebook.utils import make_sweep
from zcu_tools.program.v2 import (
    Delay,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    Readout,
    ReadoutCfg,
    Repeat,
    Reset,
    ResetCfg,
    SoftDelay,
    SweepCfg,
    sweep2param,
)
from zcu_tools.utils.datasaver import LabberPayload
from zcu_tools.utils.fitting import fit_decay, fit_decay_fringe
from zcu_tools.utils.process import rotate2real


def cpmg_signal2real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    real_signals = rotate2real(signals).real
    max_vals = np.max(real_signals, axis=1, keepdims=True)
    min_vals = np.min(real_signals, axis=1, keepdims=True)
    return (real_signals - min_vals) / np.clip(max_vals - min_vals, 1e-12, None)


@dataclass(frozen=True)
class CPMG_Result:
    ns: NDArray[np.int64]
    delays: NDArray[np.float64]
    signals: NDArray[np.complex128]
    cfg_snapshot: CPMG_Cfg | None = None


class CPMG_ModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    pi2_pulse: PulseCfg
    pi_pulse: PulseCfg
    readout: ReadoutCfg


class CPMG_SweepCfg(ConfigBase):
    times: SweepCfg | list[int]
    length: SweepCfg | None = None


class CPMG_Cfg(ProgramV2Cfg, ExpCfgModel):
    modules: CPMG_ModuleCfg
    sweep: CPMG_SweepCfg
    length_range: list[tuple[float, float]] | tuple[float, float]
    length_expts: int


CPMG_LENGTHS_ROLE = "lengths"
CPMG_SIGNALS_ROLE = "signals"
CPMG_GROUPED_ROLES = (CPMG_LENGTHS_ROLE, CPMG_SIGNALS_ROLE)


def cpmg_result_to_grouped_payloads(result: CPMG_Result) -> dict[str, LabberPayload]:
    return CPMG_GROUPED_AXES_SPEC.payloads_from_result(result)


def save_cpmg_grouped_result(
    filepath: str,
    result: CPMG_Result,
    *,
    comment: str = "",
    tag: str = "twotone/ge/cpmg",
) -> str:
    return CPMG_GROUPED_AXES_SPEC.save_grouped_result(
        filepath,
        result,
        comment=comment,
        tag=tag,
    )


def load_cpmg_grouped_result(filepath: str) -> CPMG_Result:
    return CPMG_GROUPED_AXES_SPEC.load_result(filepath)


def _validate_cpmg_arrays(
    times: NDArray[np.int64],
    lengths: NDArray[np.float64],
    signals: NDArray[np.complex128],
) -> None:
    if times.ndim != 1:
        raise ValueError(f"CPMG ns must be 1-D, got shape {times.shape}")
    if lengths.ndim != 2:
        raise ValueError(f"CPMG delays must be 2-D, got shape {lengths.shape}")
    if signals.shape != lengths.shape:
        raise ValueError(
            "CPMG signals shape must match delays shape "
            f"(got signals={signals.shape}, delays={lengths.shape})"
        )
    if lengths.shape[0] != times.shape[0]:
        raise ValueError(
            "CPMG delays first dimension must match ns length "
            f"(got delays={lengths.shape}, ns={times.shape})"
        )


def _validate_cpmg_result(result: CPMG_Result) -> None:
    _validate_cpmg_arrays(
        np.asarray(result.ns, dtype=np.int64),
        np.asarray(result.delays, dtype=np.float64),
        np.asarray(result.signals, dtype=np.complex128),
    )


def _build_cpmg_result(data: GroupedLoadData[CPMG_Cfg]) -> CPMG_Result:
    lengths_role = data.role(CPMG_LENGTHS_ROLE)
    signals_role = data.role(CPMG_SIGNALS_ROLE)
    ns = lengths_role.axes[1].astype(np.int64)
    delays = lengths_role.z.astype(np.float64)
    signals = signals_role.z.astype(np.complex128)
    _validate_cpmg_arrays(ns, delays, signals)
    return CPMG_Result(
        ns=ns,
        delays=delays,
        signals=signals,
        cfg_snapshot=data.cfg_snapshot,
    )


_CPMG_ROLE_AXES = (
    RoleAxisSpec.generated_arange("Time Index", "a.u.", dtype=np.int64),
    RoleAxisSpec("Number of Pi", "a.u.", field_name="ns", dtype=np.int64),
)
CPMG_GROUPED_AXES_SPEC = GroupedAxesSpec(
    roles=(
        RoleSpec(
            role=CPMG_LENGTHS_ROLE,
            axes=_CPMG_ROLE_AXES,
            z=RoleZSpec(
                field_name="delays",
                label="Length",
                unit="s",
                scale=US_TO_S,
                dtype=np.float64,
            ),
        ),
        RoleSpec(
            role=CPMG_SIGNALS_ROLE,
            axes=_CPMG_ROLE_AXES,
            z=RoleZSpec(
                field_name="signals",
                label="Signal",
                unit="a.u.",
                dtype=np.complex128,
            ),
        ),
    ),
    result_type=CPMG_Result,
    cfg_type=CPMG_Cfg,
    tag="twotone/ge/cpmg",
    result_builder=_build_cpmg_result,
    result_validator=_validate_cpmg_result,
)


class CPMG_Exp(AbsExperiment[CPMG_Result, CPMG_Cfg]):
    def run(
        self,
        soc,
        soccfg,
        cfg: CPMG_Cfg,
        *,
        detune_ratio: float = 0.0,
        earlystop_snr: float | None = None,
        acquire_kwargs: dict[str, Any] | None = None,
    ) -> CPMG_Result:
        orig_cfg = deepcopy(cfg)

        setup_devices(cfg, progress=True)
        modules = cfg.modules

        pi2_pulse = modules.pi2_pulse
        pi_pulse = modules.pi_pulse

        time_sweep = cfg.sweep.times
        times = sweep2array(time_sweep, allow_array=True)
        if not np.allclose(times, np.round(times)):
            raise ValueError("Times must be integers")
        times = times.astype(np.int64)

        length_ranges = cfg.length_range
        if isinstance(length_ranges, tuple):
            ranges = np.zeros((len(times), 2), dtype=np.float64)
            ranges[:, 0] = length_ranges[0]
            ranges[:, 1] = length_ranges[1]
        else:
            ranges = np.array(length_ranges, dtype=np.float64)
        length_ranges = ranges

        lengths = np.array(  # (times x length)
            [
                sweep2array(
                    make_sweep(
                        start=length_ranges[i, 0],
                        stop=length_ranges[i, 1],
                        expts=cfg.length_expts,
                    ),
                    "time",
                    {"soccfg": soccfg, "scaler": 1 / (2 * t)},
                )
                for i, t in enumerate(times)
            ]
        )
        length_idxs = np.arange(cfg.length_expts, dtype=np.int64)

        if np.min(times) <= 0:
            raise ValueError("times should be larger than 0")

        min_interval = np.min(length_ranges[:, 0] / (2 * times))
        if min_interval < 0.5 * (  # the first and last delays are half of the interval
            pi2_pulse.waveform.length + pi_pulse.waveform.length
        ):
            raise ValueError(
                "The interval is too short to measure the CPMG signal",
                f"min_interval: {min_interval:.2g}, ",
                f"pi2_pulse_length: {pi2_pulse.waveform.length:.2g}, ",
                f"pi_pulse_length: {pi_pulse.waveform.length:.2g}",
            )

        current_snr = 0.0

        def measure_fn(
            ctx: TaskState[NDArray[np.complex128], Any, CPMG_Cfg],
            update_hook: Callable[[int, list[NDArray[np.float64]]], None] | None,
        ) -> list[NDArray[np.float64]]:
            nonlocal current_snr

            cfg = ctx.cfg
            modules = cfg.modules

            assert update_hook is not None

            time = ctx.env["time"]
            pi2_pulse = modules.pi2_pulse
            pi_pulse = modules.pi_pulse
            dpulse_len = pi_pulse.waveform.length - pi2_pulse.waveform.length

            length_sweep = cfg.sweep.length
            assert length_sweep is not None
            length_param = sweep2param("length", length_sweep)
            detune_param = (
                360
                * detune_ratio
                * sweep2param(
                    "length",
                    make_sweep(start=0, step=1, expts=length_sweep.expts),
                )
            )

            interval = length_param / (2 * time)

            def update_snr(snr: float) -> None:
                nonlocal current_snr
                current_snr = snr

            return ModularProgramV2(
                soccfg,
                cfg,
                modules=[
                    Reset("reset", modules.reset),
                    Pulse("pi2_pulse1", pi2_pulse, block_mode=False),
                    Delay("first_delay", interval - 0.5 * dpulse_len),
                    Repeat("pi_loop", time - 1).add_content(
                        [
                            Pulse("pi_pulse", pi_pulse, block_mode=False),
                            SoftDelay("inner_delay", 2 * interval),
                        ]
                    ),
                    Pulse("last_pi_pulse", pi_pulse, block_mode=False),
                    Delay("last_delay", interval + 0.5 * dpulse_len),
                    Pulse(
                        name="pi2_pulse2",
                        cfg=pi2_pulse.with_updates(
                            phase=pi2_pulse.phase + detune_param
                        ),
                    ),
                    Readout("readout", modules.readout),
                ],
                sweep=[("length", length_sweep)],
            ).acquire(
                soc,
                progress=False,
                round_hook=update_hook,
                stop_checkers=[
                    ctx.is_stop,
                    snr_checker(
                        ctx,
                        earlystop_snr,
                        lambda x: rotate2real(x).real,
                        after_check=update_snr,
                    ),
                ],
                **(acquire_kwargs or {}),
            )

        with LivePlot2DwithLine(
            "Number of Pi", "Time idxs", line_axis=1, num_lines=2
        ) as viewer:

            def update_fn(
                i: int, ctx: TaskState[Any, Any, CPMG_Cfg], time: float
            ) -> None:
                ctx.env.update(time=int(time), length_idx=i)
                ctx.cfg.sweep.length = make_sweep(
                    start=length_ranges[i, 0],
                    stop=length_ranges[i, 1],
                    expts=ctx.cfg.length_expts,
                )

            signals = run_task(
                task=Task(
                    measure_fn=measure_fn,
                    result_shape=(len(length_idxs),),
                    pbar_n=cfg.rounds,
                ).scan("times", times.tolist(), before_each=update_fn),
                init_cfg=cfg,
                on_update=lambda ctx: viewer.update(
                    times.astype(np.float64),
                    lengths[ctx.env["length_idx"]],
                    cpmg_signal2real(np.asarray(ctx.root_data)),
                    title=f"snr = {current_snr:.1f}" if current_snr else None,
                ),
            )
            signals = np.asarray(signals)

        # record result
        self.last_result = CPMG_Result(
            ns=times, delays=lengths, signals=signals, cfg_snapshot=orig_cfg
        )

        return self.last_result

    def analyze(
        self,
        result: CPMG_Result | None = None,
        fit_fringe: bool = True,
        t2r: float | None = None,
        t1: float | None = None,
    ) -> Figure:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"

        times, lengths, signals2D = result.ns, result.delays, result.signals

        real_signals2D = rotate2real(signals2D).real
        norm_signals = cpmg_signal2real(signals2D)

        def is_good_fit(lens, fit_signal, real_signal, t2r, t2err) -> bool:
            fit_ptp = np.ptp(fit_signal)
            residual = real_signal - fit_signal
            smooth_residual = gaussian_filter1d(residual, sigma=1)
            mean_residual = np.mean(np.abs(smooth_residual))
            return (
                t2r < np.max(lens) and t2err < 0.5 * t2r and fit_ptp > 4 * mean_residual
            )

        t2s = np.full(len(times), np.nan, dtype=np.float64)
        t2errs = np.zeros_like(t2s)
        for i, (real_signal, lens) in enumerate(zip(real_signals2D, lengths)):
            if np.any(np.isnan(real_signal)):
                continue

            plens = lens - lens[0]
            if fit_fringe:
                t2, t2err, _, _, fit_signal, _ = fit_decay_fringe(plens, real_signal)
            else:
                t2, t2err, fit_signal, _ = fit_decay(plens, real_signal)

            if is_good_fit(plens, fit_signal, real_signal, t2, t2err):
                t2s[i] = t2
                t2errs[i] = t2err

        if np.all(np.isnan(t2s)):
            raise ValueError("No valid Fitting T2 found. Please check the data.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        assert isinstance(ax1, Axes)
        assert isinstance(ax2, Axes)

        X = np.broadcast_to(times[None, :], norm_signals.T.shape)
        Y = lengths.T
        ax1.pcolormesh(X, Y, norm_signals.T, shading="nearest")
        ax1.set_xlabel("Number of Pi")
        ax1.set_ylabel("Time (us)")

        ax2.errorbar(times, t2s, yerr=t2errs, label=r"$T_{CPMG}$", color="red", fmt=".")
        if t1 is not None:
            ax2.axhline(2 * t1, color="blue", linestyle="--", label=r"$2T_1$")
        if t2r is not None:
            ax2.axhline(t2r, color="green", linestyle="--", label=r"$T_{2r}$")
        ax2.set_xlim(0.3, None)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.set_xlabel("Number of Pi")

        fig.tight_layout()

        return fig

    def save(
        self,
        filepath: str,
        result: CPMG_Result | None = None,
        comment: str | None = None,
        tag: str = "twotone/ge/cpmg",
    ) -> None:
        if result is None:
            result = self.last_result
        assert result is not None, "no result found"
        CPMG_GROUPED_AXES_SPEC.save_experiment_result(
            filepath,
            result,
            comment=comment,
            tag=tag,
            make_comment_fn=make_comment,
        )

    def load(self, filepath: str) -> CPMG_Result:
        self.last_result = load_cpmg_grouped_result(filepath)
        return self.last_result
