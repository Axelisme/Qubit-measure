from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    IDENTITY,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
    config,
    record_result,
    retrieve_result,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.utils import setup_devices
from zcu_tools.experiment.v2.runner import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils import sweep2array
from zcu_tools.liveplot import LivePlot1D
from zcu_tools.program.base import StoppedPartialAcquireError
from zcu_tools.program.v2 import (
    DirectReadoutCfg,
    ProgramV2Cfg,
    PulseCfg,
    PulseReadoutCfg,
    ReadoutCfg,
    ResetCfg,
    SweepCfg,
)
from zcu_tools.utils.fitting.singleshot import transition_state_bin_probabilities

from .rabi_fit import RabiJointFitResult, fit_rabi_joint
from .util import classify_result, raw_shots_to_signal


@dataclass(frozen=True)
class AmpRabiResult:
    gains: NDArray[np.float64]
    shot_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]
    cfg_snapshot: AmpRabiCfg | None = None


def classify_rabi_iq(
    signals: NDArray[np.complex128],
    g_center: complex,
    e_center: complex,
    radius: float,
) -> NDArray[np.float64]:
    if signals.ndim != 2:
        raise ValueError("Amp Rabi raw IQ must have shape (length, shot)")
    g_mask, e_mask, _ = classify_result(signals, g_center, e_center, radius)
    return np.stack((g_mask.mean(axis=1), e_mask.mean(axis=1)), axis=1)


def _effective_t1_us(
    readout: ReadoutCfg | None,
    length_ratio: float,
) -> float | None:
    if readout is None or not np.isfinite(length_ratio) or length_ratio <= 0.0:
        return None
    if isinstance(readout, PulseReadoutCfg):
        readout_length = readout.ro_cfg.ro_length
    elif isinstance(readout, DirectReadoutCfg):
        readout_length = readout.ro_length
    else:
        return None
    if not isinstance(readout_length, Real):
        return None
    value = float(readout_length) / length_ratio
    return value if np.isfinite(value) and value > 0.0 else None


def _plot_initial_histogram(
    ax: Axes,
    fit: RabiJointFitResult,
    effective_t1_us: float | None,
    classified_distribution: NDArray[np.float64] | None,
) -> None:
    counts = fit.projection.counts[0]
    edges = fit.projection.bin_edges
    ax.stairs(counts, edges, label="Observed counts", color="black")

    fitted_values = np.array(
        [
            fit.projected_g_center,
            fit.projected_e_center,
            fit.sigma,
            fit.p_avg,
            fit.length_ratio,
            fit.fitted_populations[0, 0],
            fit.fitted_populations[0, 1],
        ]
    )
    if fit.backend.valid and np.isfinite(fitted_values).all():
        qg, qe = transition_state_bin_probabilities(
            edges,
            fit.projected_g_center,
            fit.projected_e_center,
            fit.sigma,
            fit.p_avg,
            fit.length_ratio,
        )
        p_g, p_e = fit.fitted_populations[0, :2]
        shots = float(counts.sum())
        g_counts = shots * p_g * qg
        e_counts = shots * p_e * qe
        ax.stairs(g_counts + e_counts, edges, label="Fitted total", color="purple")
        ax.stairs(
            g_counts,
            edges,
            label="Ground contribution",
            color="blue",
            linestyle="--",
        )
        ax.stairs(
            e_counts,
            edges,
            label="Excited contribution",
            color="red",
            linestyle="--",
        )
        title = "Initial-point projected IQ histogram"
        if effective_t1_us is not None:
            title += f"\nEffective $T_1$ = {effective_t1_us:.3g} μs"
        ax.set_title(title, fontsize=10)
        if (
            classified_distribution is not None
            and classified_distribution.shape == (3,)
            and np.isfinite(classified_distribution).all()
        ):
            g_population, e_population, l_population = classified_distribution
            ax.text(
                0.98,
                0.96,
                "Classified G/E/L\n"
                f"G {g_population:.1%}\n"
                f"E {e_population:.1%}\n"
                f"L {l_population:.1%}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "#c9cdd3", "alpha": 0.9},
            )
    else:
        ax.set_title(
            "Initial-point projected IQ histogram\nJoint fit invalid",
            fontsize=10,
        )
    ax.set_xlabel("Pooled PCA projection (a.u.)")
    ax.set_ylabel("Counts")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True)


def _plot_confusion_matrix(ax: Axes, fit: RabiJointFitResult) -> None:
    labels = ("g", "e", "other")
    matrix = fit.confusion_matrix
    if fit.backend.valid and np.isfinite(matrix).all():
        ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        for row in range(3):
            for column in range(3):
                value = matrix[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=8,
                )
        ax.set_title("Fitted confusion matrix\nOther row fixed by model", fontsize=10)
    else:
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(2.5, -0.5)
        ax.text(1.0, 1.0, "Joint fit invalid", ha="center", va="center")
        ax.set_title("Confusion matrix unavailable", fontsize=10)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Classified state")
    ax.set_ylabel("True state")


class AmpRabiSweepCfg(ConfigBase):
    gain: SweepCfg


class AmpRabiModuleCfg(ConfigBase):
    reset: ResetCfg | None = None
    qub_pulse: PulseCfg
    readout: ReadoutCfg


class AmpRabiCfg(ProgramV2Cfg, ExpCfgModel):
    modules: AmpRabiModuleCfg
    sweep: AmpRabiSweepCfg
    shots: int


class AmpRabiExp(PersistableExperiment[AmpRabiResult, AmpRabiCfg]):
    AXES_SPEC = AxesSpec(
        axes=(
            Axis(
                "shot_indices",
                "Shot Index",
                "None",
                scale=IDENTITY,
                dtype=np.int64,
            ),
            Axis("gains", "Gain", "a.u.", scale=IDENTITY, dtype=np.float64),
        ),
        z=ZSpec("signals", "Signal", "a.u.", dtype=np.complex128),
        result_type=AmpRabiResult,
        cfg_type=AmpRabiCfg,
        tag="singleshot/amp_rabi",
    )

    @record_result
    def run(
        self,
        soc,
        soccfg,
        cfg: AmpRabiCfg,
        g_center: complex,
        e_center: complex,
        radius: float,
    ) -> AmpRabiResult:
        cfg = deepcopy(cfg)
        setup_devices(cfg, progress=True)
        if cfg.rounds != 1:
            warnings.warn("rounds will be overwritten to 1 for singleshot measurement")
            cfg.rounds = 1
        if cfg.reps != cfg.shots:
            warnings.warn("reps will be overwritten by singleshot measurement shots")
            cfg.reps = cfg.shots

        modules = cfg.modules
        gains = sweep2array(
            cfg.sweep.gain,
            "gain",
            {"soccfg": soccfg, "gen_ch": modules.qub_pulse.ch},
        )
        expected_shape = (len(gains), cfg.shots)

        with LivePlot1D(
            "Gain (a.u.)",
            "Signal",
            segment_kwargs=dict(
                num_lines=3,
                line_kwargs=[
                    dict(label="Ground"),
                    dict(label="Excited"),
                    dict(label="Other"),
                ],
            ),
        ) as viewer:
            viewer.get_ax().set_ylim(0.0, 1.0)

            def update_view(raw_iq: NDArray[np.complex128]) -> None:
                acquired_rows = np.all(np.isfinite(raw_iq), axis=1)
                populations = classify_rabi_iq(raw_iq, g_center, e_center, radius)
                populations[~acquired_rows] = np.nan
                other = 1.0 - populations.sum(axis=1)
                viewer.update(gains, np.column_stack((populations, other)).T)

            buffer = SignalBuffer(
                expected_shape,
                dtype=np.complex128,
                on_update=update_view,
            )
            with Schedule(cfg, buffer) as sched:
                for gain, step in sched.scan("gain", gains.tolist()):
                    modules = step.cfg.modules
                    modules.qub_pulse.set_param("gain", gain)
                    program = (
                        step.prog_builder(soc, soccfg)
                        .add_reset("reset", modules.reset)
                        .add_pulse("qubit_pulse", modules.qub_pulse)
                        .add_readout("readout", modules.readout)
                        .build()
                    )
                    try:
                        program.acquire(soc, progress=True, cancel_flag=step.stop)
                    except StoppedPartialAcquireError:
                        step.set_stop()
                        break

                    raw_iq = raw_shots_to_signal(program)
                    expected_point_shape = (cfg.shots,)
                    if raw_iq.shape != expected_point_shape:
                        raise ValueError(
                            "Amp Rabi raw IQ shape mismatch: "
                            f"expected {expected_point_shape}, got {raw_iq.shape}"
                        )
                    buffer[step].set(raw_iq)
            signals = buffer.array
            update_view(signals)

        return AmpRabiResult(
            gains=gains,
            shot_indices=np.arange(cfg.shots, dtype=np.int64),
            signals=signals,
            cfg_snapshot=cfg,
        )

    @retrieve_result
    def analyze(
        self,
        result: AmpRabiResult | None = None,
        *,
        max_calls: int | None = None,
    ) -> tuple[RabiJointFitResult, Figure]:
        assert result is not None, "no result found"

        fit = fit_rabi_joint(result.gains, result.signals, max_calls=max_calls)
        width, height = config.figsize
        fig = plt.figure(figsize=(width, height * 1.6), layout="constrained")
        assert isinstance(fig, Figure)
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=(2.0, 1.2),
            hspace=0.16,
            wspace=0.12,
        )
        ax = fig.add_subplot(grid[0, :])
        histogram_ax = fig.add_subplot(grid[1, 0])
        confusion_ax = fig.add_subplot(grid[1, 1])

        colors = ("blue", "red", "green")
        labels = ("$|0\\rangle$", "$|1\\rangle$", "$|L\\rangle$")
        for index, (color, label) in enumerate(zip(colors, labels, strict=True)):
            ax.plot(
                result.gains,
                fit.measured_populations[:, index],
                color=color,
                ls="none",
                marker="o",
                markersize=3,
                label=label,
            )
            ax.plot(
                result.gains,
                fit.fitted_populations[:, index],
                color=color,
                ls="-",
            )
        if fit.backend.valid:
            ax.set_title(f"cond={fit.condition_number:.3g}")
        else:
            ax.set_title("Amp Rabi joint fit invalid")
        ax.set_xlabel("Pulse Gain (a.u.)")
        ax.set_ylabel("Population")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc=4)
        ax.grid(True)
        readout = (
            result.cfg_snapshot.modules.readout
            if result.cfg_snapshot is not None
            else None
        )
        effective_t1_us = (
            _effective_t1_us(readout, fit.length_ratio) if fit.backend.valid else None
        )
        classified_distribution = None
        if fit.backend.valid:
            classified_ge = classify_rabi_iq(
                result.signals[:1],
                fit.g_center,
                fit.e_center,
                fit.radius,
            )[0]
            classified_distribution = np.array(
                (*classified_ge, 1.0 - classified_ge.sum()), dtype=np.float64
            )
        _plot_initial_histogram(
            histogram_ax,
            fit,
            effective_t1_us,
            classified_distribution,
        )
        _plot_confusion_matrix(confusion_ax, fit)
        return fit, fig
