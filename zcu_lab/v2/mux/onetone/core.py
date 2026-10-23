"""Fixed-tone mux spectroscopy with a common nominal host-side offset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import StrictInt, ValidationInfo, field_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.experiment import (
    IDENTITY,
    MHZ_TO_HZ,
    AxesSpec,
    Axis,
    PersistableExperiment,
    ZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.v2.runtime.schedule import Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg, ResetCfg, SweepCfg
from zcu_tools.program.v2.modules import MuxReadout, MuxReadoutCfg


@dataclass(frozen=True)
class MuxOnetoneResult:
    """Averaged active spectra in selection order, including partial runs.

    dfs is float64 (Ndf,) of nominal common offsets in MHz, not actual DDS
    frequencies. tone_indices is int64 (K,) of original indices in 0..N-1;
    each index i refers to saved cfg.modules.readout.tones[i], not its ADC
    channel. K is the active count and N the full cfg tone count.
    signals is complex128 (Ndf, K) of averaged I+jQ in a.u., retaining the
    program's ADC-length normalization. Unwritten entries remain NaN; acquired
    rows survive cancellation or failure.
    """

    dfs: NDArray[np.float64]
    tone_indices: NDArray[np.int64]
    signals: NDArray[np.complex128]


class MuxOnetoneModuleCfg(ConfigBase):
    """Modules for one fixed-tone program.

    reset is an optional native ResetCfg, default None (no reset). readout is
    a native MuxReadoutCfg with 1..8 ordered base tones: absolute freq in MHz,
    normalized gain in [-1, 1], phase in degrees and explicit logical ro_ch.
    Its generator, mixer and shared playback/ADC timing stay fixed during the
    offset scan. MuxReadoutCfg owns validation of all readout fields.
    """

    reset: Annotated[ResetCfg | None, FieldMeta(label="Reset")] = None
    readout: Annotated[
        MuxReadoutCfg,
        FieldMeta(
            label="Readout",
            description=(
                "Without a slot, mixer_freq comes from $mux.mixer_freq "
                "independently of kappa; missing mixer stays required and blank. "
                "ro_length=1.0 us; trig_offset=$mux.time_of_fly "
                "+5/(2*pi*min group kappa); length=trig_offset+ro_length. "
                "Missing group flight time or any kappa leaves trigger and length "
                "required and blank. "
                "Candidates are not device-confirmed (未經裝置確認); "
                "capability checks run at Run. With a slot, keep its mixer and "
                "ADC window, add ringup to its trigger and extend pulse coverage."
            ),
        ),
    ]


class MuxOnetoneSweepCfg(ConfigBase):
    """Common nominal frequency-offset scan, without hardware-grid rounding.

    df is SweepCfg(start, stop, expts, step) in MHz, shared by every tone.
    start, stop and step must be finite. SweepCfg requires positive expts and
    stop = start + step * (expts - 1); one point requires start = stop, step = 0.
    Invalid values raise Pydantic validation errors at construction/assignment.
    """

    df: Annotated[SweepCfg, FieldMeta(label="Frequency offset", unit="MHz")]

    @field_validator("df")
    @classmethod
    def _validate_finite_df(cls, value: SweepCfg) -> SweepCfg:
        if not all(math.isfinite(v) for v in (value.start, value.stop, value.step)):
            raise ValueError("df start, stop and step must be finite MHz values")
        return value


class MuxOnetoneCfg(ProgramV2Cfg, ExpCfgModel):
    """Scan every selected readout tone by the same frequency offset.

    modules supplies the base mux readout and optional reset. sweep.df supplies
    nominal offsets in MHz; each point uses base_freq[i] + df with no cumulative
    update. Each point's program quantizes these frequencies to the hardware grid.

    Inherited reps (default 1) is repetitions per round; rounds (default 1) is
    software acquisition rounds per offset. initial_delay and relax_delay are
    program delays in us (both default 1). No external devices are required.
    participants selects original readout indices in execution/column order.
    Lists must be nonempty, unique integers in 0..N-1;
    bool, floating point and string indices are rejected without coercion.
    Saved data retains the full base readout, including inactive tones.
    """

    modules: MuxOnetoneModuleCfg
    sweep: MuxOnetoneSweepCfg
    participants: Annotated[
        list[StrictInt],
        FieldMeta(
            label="Participants",
            description="Original tone indices in execution/column order.",
        ),
    ]

    @field_validator("participants")
    @classmethod
    def validate_participants(
        cls, value: list[int], info: ValidationInfo
    ) -> list[int]:
        """Reject empty, duplicate or out-of-range selections; preserve order.

        Pydantic validates strict integer elements before this check. Range is
        checked against the full readout when its own validation succeeded.
        """
        if not value or len(set(value)) != len(value):
            raise ValueError("participants must be nonempty and contain unique indices")
        modules = info.data.get("modules")
        if isinstance(modules, MuxOnetoneModuleCfg) and any(
            index < 0 or index >= len(modules.readout.tones) for index in value
        ):
            raise ValueError(
                "participants indices must be in 0..N-1 of the full readout"
            )
        return value


def _mux_signal(
    prog: ModularProgramV2, acquired: Sequence[NDArray[np.float64]]
) -> NDArray[np.complex128]:
    # Builder deep-copies modules; only the program-owned instance is initialized.
    for module in prog.modules:
        if module.name == "readout" and isinstance(module, MuxReadout):
            iq = module.collect_iq(prog, acquired, read_index=0)
            return iq[:, 0] + 1j * iq[:, 1]
    raise ValueError("Mux program is missing its named readout MuxReadout")


class MuxOnetoneExp(PersistableExperiment[MuxOnetoneResult, MuxOnetoneCfg]):
    """Measure several resonator spectra with a common frequency-offset scan.

    Each point applies the optional reset, then plays and acquires the selected
    mux readout tones together. Results contain averaged complex IQ with shape
    (Ndf, K), where K is the selected tone count. Saved axes retain original tone
    indices and nominal offsets (MHz in memory, Hz on disk).
    """

    AXES_SPEC = AxesSpec(
        axes=(
            Axis("tone_indices", "Tone", "", scale=IDENTITY, dtype=np.int64),
            Axis("dfs", "Frequency offset", "Hz", scale=MHZ_TO_HZ),
        ),
        z=ZSpec("signals", "Signal", "a.u."),
        result_type=MuxOnetoneResult,
        cfg_type=MuxOnetoneCfg,
        cfg_schema_version="2.0",
        tag="mux/onetone",
    )

    def run(self, config: MuxOnetoneCfg, *, context: RunContext) -> MuxOnetoneResult:
        """Reset, read all selected tones together, and repeat at each offset.

        config supplies the base readout and common MHz offsets; it is unchanged.
        context supplies connected soc/soccfg, plots and cancellation. The live
        plot shows one magnitude curve per selected tone. Returned complex IQ is
        averaged over reps and rounds, in participants order; unmeasured points
        stay NaN after a stop. Call context.scope.stop.raise_if_error() to
        distinguish acquisition failure from a successful or stopped return.
        Invalid cfg or program settings raise.
        """
        # 1. Select tones without changing their original indices or base frequencies.
        cfg = deepcopy(config)
        base_readout = cfg.modules.readout
        dfs = sweep2array(cfg.sweep.df, "none")
        indices = cfg.participants
        tone_indices = np.asarray(indices, dtype=np.int64)
        active_readout = base_readout.with_updates(
            tones=[base_readout.tones[index] for index in indices]
        )
        # 2. Give each selected spectrum its own figure and original tone identity.
        viewers = [
            context.plots.liveplot_1d(
                f"measurement_{index}",
                "Nominal frequency offset (MHz)",
                "Magnitude (a.u.)",
                title=f"{index}: {base_readout.tones[index].label or f'tone {index}'}",
            )
            for index in indices
        ]

        def update(data: NDArray[np.complex128]) -> None:
            for column, viewer in enumerate(viewers):
                viewer.update(dfs, np.abs(data[:, column]))

        signals_buffer = SignalBuffer((len(dfs), len(tone_indices)), on_update=update)

        # 3. Rebuild reset -> mux readout at each offset, always from the base tones.
        with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
            for df, step in sched.scan("df", dfs.tolist()):
                readout = active_readout.with_updates(
                    tones=[
                        tone.with_updates(freq=tone.freq + df)
                        for tone in active_readout.tones
                    ]
                )
                builder = (
                    step.prog_builder(context.soc, context.soccfg)
                    .add_reset("reset", step.cfg.modules.reset)
                    .add(MuxReadout("readout", readout))
                )
                prog = builder.build()
                # Collect all selected ADCs and average reps/rounds into this scan row.
                _ = builder.run_program(prog, raw2signal_fn=partial(_mux_signal, prog))

        # 4. Keep acquired rows and the original tone mapping, including partial scans.
        signals_buffer.trigger_update(flush=True)
        return MuxOnetoneResult(
            dfs=dfs,
            tone_indices=tone_indices,
            signals=signals_buffer.array,
        )
