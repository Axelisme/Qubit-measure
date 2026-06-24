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
from zcu_tools.experiment import AbsExperiment
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import make_comment, parse_comment, setup_devices
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
from zcu_tools.utils.datasaver import (
    DatasetRole,
    LabberMetadata,
    LabberPayload,
    load_grouped_labber_data,
    save_grouped_labber_data,
)
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
    times = np.asarray(result.ns, dtype=np.int64)
    lengths = np.asarray(result.delays, dtype=np.float64)
    signals = np.asarray(result.signals, dtype=np.complex128)

    _validate_cpmg_arrays(times, lengths, signals)
    axes = _cpmg_grouped_axes(times, lengths)

    return {
        CPMG_LENGTHS_ROLE: LabberPayload(
            ("Length", "s", 1e-6 * lengths),
            axes=axes,
        ),
        CPMG_SIGNALS_ROLE: LabberPayload(
            ("Signal", "a.u.", signals),
            axes=axes,
        ),
    }


def save_cpmg_grouped_result(
    filepath: str,
    result: CPMG_Result,
    *,
    comment: str = "",
    tag: str = "twotone/ge/cpmg",
) -> str:
    return save_grouped_labber_data(
        filepath,
        cpmg_result_to_grouped_payloads(result),
        metadata=LabberMetadata(comment=comment, tags=tag),
    )


def load_cpmg_grouped_result(filepath: str) -> CPMG_Result:
    grouped = load_grouped_labber_data(filepath, required_roles=CPMG_GROUPED_ROLES)
    lengths_payload = grouped.roles[DatasetRole(CPMG_LENGTHS_ROLE)]
    signals_payload = grouped.roles[DatasetRole(CPMG_SIGNALS_ROLE)]

    _validate_cpmg_payload_axes(lengths_payload, signals_payload)
    ns = _payload_number_of_pi(lengths_payload)
    delays = _payload_lengths_us(lengths_payload)
    signals = np.asarray(signals_payload.z, dtype=np.complex128)
    _validate_cpmg_arrays(ns, delays, signals)

    cfg_snapshot = None
    comment = grouped.metadata.comment
    if comment:
        cfg, _, _ = parse_comment(comment)
        if cfg is not None:
            cfg_snapshot = CPMG_Cfg.validate_or_warn(cfg, source=filepath)

    return CPMG_Result(
        ns=ns,
        delays=delays,
        signals=signals,
        cfg_snapshot=cfg_snapshot,
    )


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


def _cpmg_grouped_axes(
    times: NDArray[np.int64],
    lengths: NDArray[np.float64],
) -> list[tuple[str, str, NDArray[np.float64]]]:
    time_indices = np.arange(lengths.shape[1], dtype=np.float64)
    number_of_pi = times.astype(np.float64)
    return [
        ("Time Index", "a.u.", time_indices),
        ("Number of Pi", "a.u.", number_of_pi),
    ]


def _validate_cpmg_payload_axes(
    lengths_payload: LabberPayload,
    signals_payload: LabberPayload,
) -> None:
    _require_cpmg_axes(lengths_payload, CPMG_LENGTHS_ROLE)
    _require_cpmg_axes(signals_payload, CPMG_SIGNALS_ROLE)
    for index in range(2):
        lengths_axis = lengths_payload.axes[index]
        signals_axis = signals_payload.axes[index]
        if lengths_axis.name != signals_axis.name:
            raise ValueError(
                "CPMG grouped roles must share axis names "
                f"(axis {index}: {lengths_axis.name!r} != {signals_axis.name!r})"
            )
        if lengths_axis.unit != signals_axis.unit:
            raise ValueError(
                "CPMG grouped roles must share axis units "
                f"(axis {index}: {lengths_axis.unit!r} != {signals_axis.unit!r})"
            )
        if not np.array_equal(
            np.asarray(lengths_axis.values), np.asarray(signals_axis.values)
        ):
            raise ValueError(f"CPMG grouped roles axis {index} values differ")


def _require_cpmg_axes(payload: LabberPayload, role: str) -> None:
    if len(payload.axes) != 2:
        raise ValueError(f"CPMG {role!r} role must have exactly two axes")
    expected = (("Time Index", "a.u."), ("Number of Pi", "a.u."))
    for index, (expected_name, expected_unit) in enumerate(expected):
        axis = payload.axes[index]
        if axis.name != expected_name or axis.unit != expected_unit:
            raise ValueError(
                f"CPMG {role!r} axis {index} must be "
                f"{expected_name!r} / {expected_unit!r}, got "
                f"{axis.name!r} / {axis.unit!r}"
            )


def _payload_number_of_pi(payload: LabberPayload) -> NDArray[np.int64]:
    number_of_pi = np.asarray(payload.axes[1].values, dtype=np.float64)
    rounded = np.round(number_of_pi)
    if not np.allclose(number_of_pi, rounded):
        raise ValueError("CPMG Number of Pi axis must contain integer values")
    return rounded.astype(np.int64)


def _payload_lengths_us(payload: LabberPayload) -> NDArray[np.float64]:
    lengths_s = np.asarray(payload.z)
    if np.iscomplexobj(lengths_s):
        if not np.allclose(np.imag(lengths_s), 0.0):
            raise ValueError("CPMG lengths role must not contain imaginary values")
        lengths_s = np.real(lengths_s)
    return np.asarray(lengths_s, dtype=np.float64) * 1e6


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

        cfg = result.cfg_snapshot
        if cfg is None:
            raise ValueError("cfg_snapshot is None")

        comment = make_comment(cfg, comment)
        save_cpmg_grouped_result(filepath, result, comment=comment, tag=tag)

    def load(self, filepath: str) -> CPMG_Result:
        self.last_result = load_cpmg_grouped_result(filepath)
        return self.last_result
