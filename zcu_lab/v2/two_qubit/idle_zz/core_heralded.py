"""Herald-conditioned idle ZZ from acquisition-side class sums."""

from dataclasses import dataclass
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
from zcu_tools.experiment.v2.utils.round_zcu import round_sweep_dict, sweep2array
from zcu_tools.program.v2 import Classify, Pulse, Reset, TimelineBuilder, shot_sums
from zcu_tools.program.v2.modules import MuxReadout
from zcu_lab.v2.support import live_line
from zcu_lab.v2.support.device import set_flux
from zcu_tools.plotting.plots import Plots

from .._sums import herald_probabilities, sigma, terminal_counts, validate_classifiers
from ._analysis import IdleBranchFit, IdleZZAnalysis, complete_points, fit_traces, sigma_errors, summarize, validate_axes
from .core import IdleZZCfg


class IdleZZHeraldedCfg(IdleZZCfg):
    """IdleZZCfg with a full mux herald before spectator preparation.

    reset=None emits no reset. Both reads use readout and its circle radii;
    readout.post_delay supplies the wait after herald. target is f1 or f2.
    """


@dataclass(frozen=True)
class IdleZZHeraldedResult:
    """Axes match IdleZZResult; sums is float64 (D,2,T,3,3,3,3).

    Class order is herald F1, herald F2, terminal F1, terminal F2, each g/e/other.
    All herald classes are saved unnormalized. Unmeasured points remain NaN.
    """

    dc_values: NDArray[np.float64]
    spectator_states: NDArray[np.int64]
    delays: NDArray[np.float64]
    sums: NDArray[np.float64]


@dataclass(frozen=True)
class IdleZZHeraldedAnalysis:
    """terminal reports joint frequencies in actual spectator g/e order.

    terminal.branches contains gg traces; ee_branches contains paired ee traces
    in actual spectator g/e order, with raw inverted fringes preserved. Each
    actual state shares frequency/decay across heralds, not A/phase/offset.

    herald_matrix is (D,2,T,3,3), all nine herald probabilities. pass_fraction
    is its (g,g) cell (D,2,T), not ground-state purity. Empty points give NaN.
    """

    terminal: IdleZZAnalysis
    ee_branches: tuple[IdleBranchFit, ...]
    herald_matrix: NDArray[np.float64]
    pass_fraction: NDArray[np.float64]


def restore_result(data: GroupedLoadData[IdleZZHeraldedCfg]) -> IdleZZHeraldedResult:
    """Restore 81-class sums; old raw-IQ files missing sums raise ValueError."""
    sums = data.variable("sums")
    return IdleZZHeraldedResult(
        sums.axes[6].astype(np.float64),
        sums.axes[5].astype(np.int64),
        sums.axes[4].astype(np.float64),
        sums.z.astype(np.float64),
    )


class IdleZZHeraldedExp(Experiment[IdleZZHeraldedResult, IdleZZHeraldedCfg]):
    """Estimate conditional fringe differences using classified gg and ee heralds."""

    AXES_SPEC: ClassVar[GroupedAxesSpec[IdleZZHeraldedResult, IdleZZHeraldedCfg]] = (
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="sums",
                    axes=(
                        VariableAxisSpec.generated_arange("terminal_f2", ""),
                        VariableAxisSpec.generated_arange("terminal_f1", ""),
                        VariableAxisSpec.generated_arange("herald_f2", ""),
                        VariableAxisSpec.generated_arange("herald_f1", ""),
                        VariableAxisSpec("Delay", "s", field_name="delays", scale=1e-6),
                        VariableAxisSpec(
                            "Spectator state (0=I, 1=Xπ)",
                            "",
                            field_name="spectator_states",
                            dtype=np.int64,
                        ),
                        VariableAxisSpec(
                            "DC source value (inactive if disabled)",
                            "a.u.",
                            field_name="dc_values",
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
            result_type=IdleZZHeraldedResult,
            cfg_type=IdleZZHeraldedCfg,
            cfg_schema_version="5.0",
            tag="two_qubit/idle_zz_heralded",
            result_builder=restore_result,
        )
    )

    def run(self, config: IdleZZHeraldedCfg, *, context: RunContext) -> IdleZZHeraldedResult:
        """Acquire reset if set → herald → spectator I/Xπ → target Ramsey → terminal.

        Hardware scans spectator_states and delay; DC alone stays on host
        because it changes an external instrument. Return unnormalized counts.
        """
        validate_classifiers(config.readout, config.roles)
        delay_sweep = round_sweep_dict(config.delay, "time", {"soccfg": context.soccfg})
        delays = sweep2array(delay_sweep)
        dc_values = np.array([0.0]) if config.dc_device is None else sweep2array(config.dc_values)
        viewer = live_line(context.plots, "measurement", "Delay (us)", "Target ⟨σ⟩ (gg, −ee; actual spectator g/e)", num_lines=len(dc_values) * 4)

        def update_view(sums: NDArray[np.float64]) -> None:
            gg = sigma(terminal_counts(sums, ("g", "g")), config.target)
            ee = -sigma(terminal_counts(sums, ("e", "e")), config.target)[:, ::-1]
            viewer.update(delays, np.stack((gg, ee), axis=1).reshape(-1, len(delays)))

        buffer = SignalBuffer((len(dc_values), 2, len(delays), 3, 3, 3, 3), dtype=np.float64, on_update=update_view)
        with Schedule(config, buffer, scope=context.scope) as sched:
            # DC stays on host because it sets an external instrument.
            for dc_value, step in sched.scan("dc", dc_values.tolist()):
                if config.dc_device is not None:
                    set_flux(context.devices[config.dc_device], dc_value, unit=config.dc_unit, stop=context.scope.stop)
                builder = TimelineBuilder(context.soccfg, reps=config.reps, initial_delay=config.initial_delay, relax_delay=config.relax_delay)
                builder.add_loop("spectator_states", 2)  # 0 = I, 1 = Xπ (outer)
                delay = builder.add_sweep("delay", delay_sweep)
                spectator_cfg = config.x180_f2 if config.target == "f1" else config.x180_f1
                pi2_cfg = config.x90_f1 if config.target == "f1" else config.x90_f2
                spectator = Pulse("spectator_x180", spectator_cfg)
                pi2 = Pulse("target_pi2", pi2_cfg)
                projection = Pulse("projection", pi2_cfg, phase=pi2_cfg.phase + 360 * config.detune * delay)
                readout = MuxReadout("readout", config.readout)
                if config.reset is not None:
                    builder.emit(Reset("reset", config.reset))
                builder.emit(readout, tag="herald")
                builder.sync()  # Herald/reset must finish before spectator preparation.
                with builder.branch(by="spectator_states", align_timeline=True) as cases:
                    with cases.case(1):
                        builder.emit(spectator)
                builder.emit(pi2)
                builder.delay_timeline(delay)
                builder.emit(projection)
                builder.emit(readout, tag="terminal")
                prog = builder.build()
                axes = [
                    Classify("herald", config.roles.f1_tone),
                    Classify("herald", config.roles.f2_tone),
                    Classify("terminal", config.roles.f1_tone),
                    Classify("terminal", config.roles.f2_tone),
                ]
                step.run_acquire_summed(prog, context.soc, processor=shot_sums(prog, axes), rounds=config.rounds, options=RunOptions())
        return IdleZZHeraldedResult(dc_values, np.arange(2, dtype=np.int64), delays, buffer.array)

    def analyze(
        self, source: RunRecord[IdleZZHeraldedCfg, IdleZZHeraldedResult],
        options: None, *, plots: Plots,
    ) -> IdleZZHeraldedAnalysis:
        """Joint-fit gg/ee by actual spectator state, then plot all herald classes.

        ee has inverted target fringes and exchanged spectator states. Full
        requested delay axes/rounds are required before herald selection.
        Shared frequency/decay determine ζ; each herald keeps local A/phase/offset.
        """
        del options
        cfg, result = source.cfg, source.result
        if cfg is None:
            raise ValueError("Idle ZZ analysis requires saved cfg")
        dc_values = np.array([0.0]) if cfg.dc_device is None else sweep2array(cfg.dc_values)
        validate_axes(result.dc_values, result.spectator_states, result.delays, result.sums, dc_values, cfg.delay.expts)
        complete = complete_points(result.sums, cfg.reps * cfg.rounds)
        gg = terminal_counts(result.sums, ("g", "g"))
        # ee starts with spectator e: nominal I is actual e, Xπ is actual g.
        ee = terminal_counts(result.sums, ("e", "e"))[:, ::-1]
        gg_traces, ee_traces = sigma(gg, cfg.target), sigma(ee, cfg.target)
        gg_fits, ee_fits = [], []
        for dc_index in range(len(dc_values)):
            for state in result.spectator_states:
                pair = (
                    IdleBranchFit(dc_index, int(state), int(np.nansum(gg[dc_index, state])), bool(np.all(complete[dc_index, state])), gg_traces[dc_index, state]),
                    IdleBranchFit(dc_index, int(state), int(np.nansum(ee[dc_index, state])), bool(np.all(complete[dc_index, 1 - state])), ee_traces[dc_index, state]),
                )
                a, b = fit_traces(pair, (sigma_errors(gg[dc_index, state], cfg.target), sigma_errors(ee[dc_index, state], cfg.target)), delays=result.delays, detune=cfg.detune)
                gg_fits.append(a)
                ee_fits.append(b)
        terminal = summarize(tuple(gg_fits))
        dc_label = "DC not swept (inactive coordinate)" if cfg.dc_device is None else f"{cfg.dc_device} ({cfg.dc_unit})"
        for dc_index, dc_value in enumerate(dc_values):
            fig, ax = plots.subplots(f"ramsey_{dc_index}", figsize=(10, 7))
            reasons = []
            for state in range(2):
                for label, fits, sign in (("gg", gg_fits, 1), ("ee", ee_fits, -1)):
                    fit = fits[2 * dc_index + state]
                    state_label = "g" if state == 0 else "e"
                    name = f"{label}, actual spectator {state_label}"
                    ax.plot(result.delays, sign * fit.trace, ".", label=f"{name}; n={fit.acquired_shots}")
                    if fit.fit_curve is not None:
                        ax.plot(result.delays, sign * fit.fit_curve, label=f"{name} joint fit |f|={fit.magnitude_mhz:.5g} MHz")
                    if fit.reason:
                        reasons.append(f"{name}: {fit.reason}")
            zeta = terminal.zeta_mhz[dc_index]
            subtitle = f"ζ={zeta:.5g} MHz" if zeta is not None else "ζ unavailable"
            ax.set(xlabel="Delay (us)", ylabel="Target ⟨σ⟩ (gg; −ee)", title=f"Target {cfg.target}; {dc_label}={dc_value:g}; {subtitle}\nShared frequency/decay by actual spectator; ee displayed inverted")
            ax.text(0.01, -0.2, "\n".join(reasons), transform=ax.transAxes, va="top", fontsize=8)
            ax.legend()
            ax.grid(True)
            fig.tight_layout()
        fig, ax = plots.subplots("zeta", figsize=(8, 5))
        for index, (value, error) in enumerate(zip(terminal.zeta_mhz, terminal.zeta_error_mhz, strict=True)):
            if value is not None:
                ax.errorbar(dc_values[index], value, yerr=error, marker="o", color="C0")
        if not any(value is not None for value in terminal.zeta_mhz):
            ax.text(0.5, 0.5, "ζ unavailable: see branch reasons", transform=ax.transAxes, ha="center")
        ax.set(xlabel=dc_label, ylabel="ζ = f_e − f_g (MHz)", title="Joint gg/ee single-frequency model; not pure Hamiltonian ZZ")
        fig.tight_layout()
        matrix = herald_probabilities(result.sums)
        labels = [f"{f1},{f2}" for f1 in ("g", "e", "other") for f2 in ("g", "e", "other")]
        for dc_index, dc_value in enumerate(dc_values):
            for state in range(2):
                fig, ax = plots.subplots(f"herald_{dc_index}_{state}", figsize=(9, 5))
                for outcome, label in enumerate(labels):
                    ax.plot(result.delays, matrix[dc_index, state].reshape(-1, 9)[:, outcome], label=label)
                ax.set(xlabel="Delay (us)", ylabel="Herald probability", title=f"DC={dc_value:g}; nominal spectator {'I' if state == 0 else 'Xπ'}", ylim=(0, 1))
                ax.legend()
                fig.suptitle("(g,g) is pass fraction, not ground-state purity")
                fig.tight_layout()
        return IdleZZHeraldedAnalysis(terminal, tuple(ee_fits), matrix, matrix[..., 0, 0])
