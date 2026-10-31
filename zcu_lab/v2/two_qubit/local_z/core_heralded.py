"""Ground-ground heralded two-target local Z."""

from dataclasses import dataclass
from itertools import product
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment import (
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.plotting.plots import Plots
from zcu_tools.program.v2 import (
    Classify,
    Pulse,
    Reset,
    SweepCfg,
    TimelineBuilder,
    shot_sums,
)
from zcu_tools.program.v2.modules import Join, MuxReadout

from zcu_lab.v2.support import live_line

from .._phase import QuadraturePhase, count_phase, phase_variance, sigma_variance
from .._sums import herald_probabilities, sigma, terminal_counts, validate_classifiers
from .analysis import LocalZAnalysis, LocalZTargetFit, fit_target
from .core import LocalZCfg


class LocalZHeraldedCfg(LocalZCfg):
    """Both local-Z targets preceded by optional reset and a mux herald.

    All fields and units follow LocalZCfg. Both reads use the same mux
    object and classifiers. readout.post_delay is the wait after herald.
    """


@dataclass(frozen=True)
class LocalZHeraldedResult:
    """Scan coordinates follow LocalZResult; sums retains all 81 classes.

    sums is float64 (2,2,Nn,3,3,3,3), ordered herald F1/F2 then terminal
    F1/F2, each g/e/other. No selection or normalization precedes persistence.
    Unmeasured points remain NaN; completed rounds survive cancellation.
    """

    targets: NDArray[np.int64]
    quadratures: NDArray[np.int64]
    gate_counts: NDArray[np.int64]
    sums: NDArray[np.float64]


def restore_result(
    data: GroupedLoadData[LocalZHeraldedCfg],
) -> LocalZHeraldedResult:
    """Restore 81-class sums; reject legacy raw-IQ files without sums."""
    sums = data.variable("sums")
    return LocalZHeraldedResult(
        targets=sums.axes[6].astype(np.int64),
        quadratures=sums.axes[5].astype(np.int64),
        gate_counts=sums.axes[4].astype(np.int64),
        sums=sums.z.astype(np.float64),
    )


@dataclass(frozen=True)
class LocalZHeraldedAnalysis(LocalZAnalysis):
    """gg writeback fits plus independent ee diagnostics in the same target order.

    ee_phases is (2,Nn), ee_quadratures and ee_acquired_shots are (2,2,Nn).
    ee_fits has one weighted fit per target. slope_difference_deg is ee−gg per
    target, or None when either fit is unavailable. No shared slope is imposed.
    """

    ee_phases: tuple[tuple[QuadraturePhase, ...], ...] = ()
    ee_quadratures: NDArray[np.float64] | None = None
    ee_acquired_shots: NDArray[np.float64] | None = None
    ee_fits: tuple[LocalZTargetFit, ...] = ()
    slope_difference_deg: tuple[float | None, ...] = ()


class LocalZHeraldedExp(Experiment[LocalZHeraldedResult, LocalZHeraldedCfg]):
    """Analyze classified X/Y projections conditioned on herald (g,g)."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[LocalZHeraldedResult, LocalZHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec.generated_arange("herald_f2", ""),
                        VariableAxisSpec.generated_arange("herald_f1", ""),
                        VariableAxisSpec(
                            "CZ gate count",
                            "",
                            field_name="gate_counts",
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            "Projection X or Y",
                            "",
                            field_name="quadratures",
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            "Target F1 or F2", "", field_name="targets", dtype=np.int64
                        ),
                    ),
                    z=VariableZSpec(
                        field_name="sums",
                        label="Herald and terminal class counts",
                        unit="",
                        dtype=np.float64,
                    ),
                ),
            ),
            result_type=LocalZHeraldedResult,
            cfg_type=LocalZHeraldedCfg,
            tag="two_qubit/local_z_heralded",
            result_builder=restore_result,
        )
    )

    def run(
        self, config: LocalZHeraldedCfg, *, context: RunContext
    ) -> LocalZHeraldedResult:
        """Acquire target/quadrature hardware loops for each host gate count.

        Counts retain target × quadrature × gate_count axes. No IQ is stored.
        cfg stays unchanged; completed rounds survive cancellation.
        """
        validate_classifiers(config.readout, config.roles)
        viewer = live_line(
            context.plots,
            "measurement",
            "CZ gate count",
            "Target sigma | herald (g,g)",
            num_lines=4,
        )

        def update_view(sums: NDArray[np.float64]) -> None:
            counts = terminal_counts(np.moveaxis(sums, 0, 2), ("g", "g"))
            values = np.stack((sigma(counts[0], "f1"), sigma(counts[1], "f2")))
            viewer.update(np.array(config.gate_counts), values.reshape(4, -1))

        buffer = SignalBuffer(
            (len(config.gate_counts), 2, 2, 3, 3, 3, 3),
            dtype=np.float64,
            on_update=update_view,
        )
        with Schedule(config, buffer, scope=context.scope) as sched:
            # gate_count changes sequence structure, so only this axis stays on host.
            for n, step in sched.scan("gate_count", config.gate_counts):
                builder = TimelineBuilder(
                    context.soccfg,
                    reps=config.reps,
                    initial_delay=config.initial_delay,
                    relax_delay=config.relax_delay,
                )
                builder.add_loop("target", 2)  # 0 = F1, 1 = F2
                quadrature = builder.add_sweep(
                    "quadrature", SweepCfg(start=0, stop=90, expts=2, step=90)
                )
                prepare_f1 = Pulse("prepare_f1", config.x90_f1)
                prepare_f2 = Pulse("prepare_f2", config.x90_f2)
                projection_f1 = Pulse(
                    "projection_f1",
                    config.x90_f1,
                    phase=config.x90_f1.phase + quadrature,
                )
                projection_f2 = Pulse(
                    "projection_f2",
                    config.x90_f2,
                    phase=config.x90_f2.phase + quadrature,
                )
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.emit(readout, tag="herald")
                with builder.branch(by="target", align_timeline=True) as cases:
                    with cases.case(0):
                        builder.emit(prepare_f1)
                    with cases.case(1):
                        builder.emit(prepare_f2)
                drive = Join(
                    Pulse("cz_f1", config.drive.f1),
                    Pulse("cz_f2", config.drive.f2),
                )
                for _ in range(n):
                    builder.emit(drive)
                builder.sync()  # finish the common train before target projection
                with builder.branch(by="target", align_timeline=True) as cases:
                    with cases.case(0):
                        builder.emit(projection_f1)
                    with cases.case(1):
                        builder.emit(projection_f2)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                step.run_acquire_summed(
                    prog,
                    context.soc,
                    processor=shot_sums(
                        prog,
                        [
                            Classify("herald", config.roles.f1_tone),
                            Classify("herald", config.roles.f2_tone),
                            Classify("terminal", config.roles.f1_tone),
                            Classify("terminal", config.roles.f2_tone),
                        ],
                    ),
                    rounds=config.rounds,
                    options=RunOptions(),
                )
        buffer.trigger_update(flush=True)
        return LocalZHeraldedResult(
            np.arange(2, dtype=np.int64),
            np.arange(2, dtype=np.int64),
            np.array(config.gate_counts, dtype=np.int64),
            np.moveaxis(buffer.array, 0, 2),
        )

    def analyze(
        self,
        source: RunRecord[LocalZHeraldedCfg, LocalZHeraldedResult],
        options: None,
        *,
        plots: Plots,
    ) -> LocalZHeraldedAnalysis:
        """Fit gg/ee independently and report ee-minus-gg slopes; writeback uses gg."""
        del options
        cfg, data = source.cfg, source.result
        if cfg is None:
            raise ValueError("Phase analysis requires saved source cfg")
        for actual, expected in (
            (data.targets, np.arange(2)),
            (data.quadratures, np.arange(2)),
            (data.gate_counts, np.array(cfg.gate_counts)),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError("LocalZ count axes must match saved cfg")
        if data.sums.shape != (2, 2, len(cfg.gate_counts), 3, 3, 3, 3):
            raise ValueError("Expected target × quadrature × gate_count × 3 × 3 counts")
        gg = terminal_counts(data.sums, ("g", "g"))
        ee = terminal_counts(data.sums, ("e", "e"))
        gg_quadratures = np.stack((sigma(gg[0], "f1"), sigma(gg[1], "f2")))
        # Starting the target in e reverses both projections, not their slope.
        ee_quadratures = -np.stack((sigma(ee[0], "f1"), sigma(ee[1], "f2")))
        gg_variances = np.stack(
            (sigma_variance(gg[0], "f1"), sigma_variance(gg[1], "f2"))
        )
        ee_variances = np.stack(
            (sigma_variance(ee[0], "f1"), sigma_variance(ee[1], "f2"))
        )
        gg_phases = tuple(
            tuple(
                count_phase(
                    float(gg_quadratures[t, 0, n]), float(gg_quadratures[t, 1, n])
                )
                for n in range(len(cfg.gate_counts))
            )
            for t in range(2)
        )
        ee_phases = tuple(
            tuple(
                count_phase(
                    float(ee_quadratures[t, 0, n]), float(ee_quadratures[t, 1, n])
                )
                for n in range(len(cfg.gate_counts))
            )
            for t in range(2)
        )
        gg_fits = tuple(
            fit_target(
                np.array(cfg.gate_counts, dtype=np.float64),
                gg_phases[t],
                phase_variance(
                    gg_quadratures[t, 0],
                    gg_quadratures[t, 1],
                    gg_variances[t, 0],
                    gg_variances[t, 1],
                ),
            )
            for t in range(2)
        )
        ee_fits = tuple(
            fit_target(
                np.array(cfg.gate_counts, dtype=np.float64),
                ee_phases[t],
                phase_variance(
                    ee_quadratures[t, 0],
                    ee_quadratures[t, 1],
                    ee_variances[t, 0],
                    ee_variances[t, 1],
                ),
            )
            for t in range(2)
        )
        differences = tuple(
            None
            if g.slope_deg is None or e.slope_deg is None
            else e.slope_deg - g.slope_deg
            for g, e in zip(gg_fits, ee_fits, strict=True)
        )
        herald_matrix = herald_probabilities(data.sums)
        answer = LocalZHeraldedAnalysis(
            gg_phases,
            gg_quadratures,
            gg.sum(axis=(-2, -1)),
            tuple(cfg.gate_counts),
            gg_fits,
            herald_matrix,
            herald_matrix[..., 0, 0],
            ee_phases,
            ee_quadratures,
            ee.sum(axis=(-2, -1)),
            ee_fits,
            differences,
        )

        fig, _ = plots.subplots(
            "phase", nrows=3, figsize=(10, 10), layout="constrained"
        )
        for herald, phases, fits in (
            ("gg", gg_phases, gg_fits),
            ("ee", ee_phases, ee_fits),
        ):
            for t in range(2):
                label = f"{herald} F{t + 1}"
                fig.axes[0].plot(
                    cfg.gate_counts, [p.contrast for p in phases[t]], ".-", label=label
                )
                fig.axes[1].plot(
                    cfg.gate_counts,
                    [
                        np.nan if p.logical_phase_deg is None else p.logical_phase_deg
                        for p in phases[t]
                    ],
                    ".-",
                    label=label,
                )
                fit = fits[t]
                if fit.unwrapped_deg is not None and fit.fit_curve is not None:
                    fig.axes[2].plot(
                        cfg.gate_counts, fit.unwrapped_deg, ".", label=label
                    )
                    fig.axes[2].plot(
                        cfg.gate_counts,
                        fit.fit_curve,
                        label=f"{label} slope={fit.slope_deg:.5g} ± {fit.slope_error_deg:.2g} deg",
                    )
        fig.axes[2].set_title(
            f"ee − gg slopes (deg/gate): F1={differences[0]}, F2={differences[1]}; writeback uses gg"
        )
        for ax, label in zip(
            fig.axes,
            ("Contrast", "Wrapped phase (deg)", "Unwrapped phase (deg)"),
            strict=True,
        ):
            ax.set(xlabel="CZ gate count", ylabel=label)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize="small")
        fig, axes = plots.subplots("quadratures", figsize=(10, 5))
        for (herald, values), t, (q, projection) in product(
            (("gg", gg_quadratures), ("ee sign-corrected", ee_quadratures)),
            range(2),
            enumerate(("X", "Y")),
        ):
            axes.plot(
                cfg.gate_counts,
                values[t, q],
                ".-",
                label=f"{herald} F{t + 1} {projection}",
            )
        axes.set(xlabel="CZ gate count", ylabel="Target sigma")
        axes.legend(ncols=2)
        fig, axes = plots.subplots("herald", figsize=(12, 5))
        image = axes.imshow(
            np.ma.masked_invalid(herald_matrix.reshape(-1, 9)),
            vmin=0,
            vmax=1,
            cmap="Blues",
            aspect="auto",
        )
        axes.set(
            xticks=range(9),
            xticklabels=[
                f"{a},{b}" for a in ("g", "e", "other") for b in ("g", "e", "other")
            ],
            xlabel="Herald class (F1, F2)",
            ylabel="Scan row (target, projection, n)",
            title="Herald distribution; (g,g) is pass fraction, not ground-state purity",
        )
        fig.colorbar(image, ax=axes, label="Herald probability")
        return answer
