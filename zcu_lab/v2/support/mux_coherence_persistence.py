"""Native mapping for common requested delay and fixed per-tone applied detunes."""

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    US_TO_S,
    GroupedAxesSpec,
    GroupedLoadData,
    RunValueAxis,
    RunValueSpec,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from zcu_lab.v2.mux._moments import iq_moments_variable, restore_iq_moments

from .mux_coherence import MuxCoherenceResult, validate_coherence_result


def coherence_spec[T: ConfigBase](
    cfg_type: type[T], *, tag: str
) -> GroupedAxesSpec[MuxCoherenceResult, T]:
    """Map Ramsey/Echo IQ, optional Native moments and fixed applied detunes.

    cfg_type restores the full leaf cfg; tag identifies Ramsey or Echo. Memory
    uses us/MHz, disk uses seconds/Hz. The IQ and moments axes use requested
    delay, never reconstructed from current cfg. Applied detunes align with
    the saved IQ tone order and may be unknown. Labber exports only the mean
    IQ. No I/O occurs. Invalid declarations or loaded arrays raise ValueError.
    """
    tone = VariableAxisSpec(
        label="Tone", unit="", field_name="tone_indices", dtype=np.int64
    )
    iq_axes = (
        VariableAxisSpec(
            label="Requested delay",
            unit="s",
            field_name="times",
            scale=US_TO_S,
        ),
        tone,
    )

    def build(data: GroupedLoadData[T]) -> MuxCoherenceResult:
        signals = data.variable("signals")

        return MuxCoherenceResult(
            signals.axes[1].astype(np.int64),
            signals.axes[0].astype(np.float64),
            signals.z.astype(np.complex128),
            data.run_values.vector("actual_detunes", dtype=np.float64),
            restore_iq_moments(data.variables.get(DataVariable("iq_moments")), signals),
        )

    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="signals",
                axes=iq_axes,
                z=VariableZSpec(
                    field_name="signals",
                    label="Signal",
                    unit="a.u.",
                    dtype=np.complex128,
                ),
            ),
            iq_moments_variable(iq_axes),
        ),
        result_type=MuxCoherenceResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=build,
        result_validator=validate_coherence_result,
        cfg_schema_version="3.0",
        run_values=(
            RunValueSpec(
                field_name="actual_detunes",
                wire_name="actual_detunes",
                unit="Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
                allow_unknown=True,
                axis=RunValueAxis(variable=DataVariable("signals"), label="Tone"),
            ),
        ),
        labber_layout="independent",
        labber_variables=("signals",),
    )
