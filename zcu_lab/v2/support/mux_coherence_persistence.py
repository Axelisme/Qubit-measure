"""Native mapping for common requested delay and per-tone coherence metadata."""

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    US_TO_S,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from zcu_lab.v2.mux._moments import iq_moments_variable, restore_iq_moments

from .mux_coherence import MuxCoherenceResult, validate_coherence_result


def coherence_spec[T: ConfigBase](
    cfg_type: type[T], *, tag: str
) -> GroupedAxesSpec[MuxCoherenceResult, T]:
    """Map Ramsey/Echo IQ versus delay and per-tone frequencies to saved grids.

    cfg_type restores the full leaf cfg; tag identifies Ramsey or Echo. Memory
    uses us/MHz, disk uses seconds/Hz. Labber keeps independent IQ and frequency
    grids and excludes optional Native moments. The loaded delay is never
    reconstructed from current cfg. No I/O occurs. Invalid declarations or
    mismatched grids raise ValueError via native persistence.
    """
    tone = VariableAxisSpec(
        label="Tone", unit="", field_name="tone_indices", dtype=np.int64
    )
    iq_axes = (
        VariableAxisSpec(
            label="Requested delay", unit="s", field_name="times", scale=US_TO_S
        ),
        tone,
    )

    def build(data: GroupedLoadData[T]) -> MuxCoherenceResult:
        # Align IQ and frequency metadata by the saved original tone indices.
        signals = data.variable("signals")
        detunes = data.variable("actual_detunes")
        frequencies = data.variable("source_frequencies")
        if not all(
            np.array_equal(signals.axes[1], variable.axes[0])
            for variable in (detunes, frequencies)
        ):
            raise ValueError(
                "Coherence variables must preserve the same original tone axis"
            )

        result = MuxCoherenceResult(
            signals.axes[1].astype(np.int64),
            signals.axes[0].astype(np.float64),
            signals.z.astype(np.complex128),
            detunes.z.astype(np.float64),
            frequencies.z.astype(np.float64),
            restore_iq_moments(data.variables.get(DataVariable("iq_moments")), signals),
        )
        validate_coherence_result(result)
        return result

    # Save the delay grid and per-tone metadata independently, without broadcasting.
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
            *[
                VariableSpec(
                    variable=name,
                    axes=(tone,),
                    z=VariableZSpec(
                        field_name=name,
                        label=label,
                        unit="Hz",
                        scale=MHZ_TO_HZ,
                        dtype=np.float64,
                    ),
                )
                for name, label in (
                    ("actual_detunes", "Actual detune"),
                    ("source_frequencies", "Source frequency"),
                )
            ],
            iq_moments_variable(iq_axes),
        ),
        result_type=MuxCoherenceResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=build,
        result_validator=validate_coherence_result,
        cfg_schema_version="3.0",
        labber_layout="independent",
        labber_variables=("signals", "actual_detunes", "source_frequencies"),
    )
