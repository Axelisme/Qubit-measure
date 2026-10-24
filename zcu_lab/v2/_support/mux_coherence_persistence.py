"""Native mapping for common requested delay and per-tone coherence metadata."""

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    US_TO_S,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from .mux_coherence import MuxCoherenceResult, validate_coherence_result


def coherence_spec[T: ConfigBase](
    cfg_type: type[T], *, tag: str
) -> GroupedAxesSpec[MuxCoherenceResult, T]:
    """Map Ramsey/Echo IQ versus delay and per-tone frequencies to saved grids.

    cfg_type restores the full leaf cfg; tag identifies Ramsey or Echo. Memory
    uses us/MHz, disk uses seconds/Hz. Labber uses independent variable grids.
    Requested delay is stored once as the IQ axis, never reconstructed from
    current cfg. No I/O occurs. Invalid declarations or mismatched loaded tone
    axes raise ValueError via native persistence.
    """
    tone = VariableAxisSpec("Tone", "", field_name="tone_indices", dtype=np.int64)

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
        )
        validate_coherence_result(result)
        return result

    # Save the delay grid and per-tone metadata independently, without broadcasting.
    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="signals",
                axes=(
                    VariableAxisSpec(
                        "Requested delay", "s", field_name="times", scale=US_TO_S
                    ),
                    tone,
                ),
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
        ),
        result_type=MuxCoherenceResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=build,
        result_validator=validate_coherence_result,
        cfg_schema_version="3.0",
        labber_layout="independent",
    )
