"""Native grouped mapping for independent zipped coordinates and averaged IQ."""

import numpy as np
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel

from .mux import MuxSweepResult, validate_mux_result


def mux_sweep_spec[T: ExpCfgModel](
    cfg_type: type[T],
    *,
    tag: str,
    coordinate_label: str,
    unit: str,
    scale: float,
) -> GroupedAxesSpec[MuxSweepResult, T]:
    """Map each tone's scan coordinates and mean IQ to matching saved grids.

    cfg_type is the leaf's typed cfg, tag its unique persistence identity.
    coordinate_label/unit are disk metadata; scale converts memory coordinates
    to disk units, e.g. MHz to Hz. Each variable is (K,P), with inner-first Point
    then Tone axes. No I/O occurs. Invalid declarations raise from GroupedAxesSpec.
    The loader checks matching axes and result contracts without consulting cfg.
    """
    # Keep each tone's own coordinates on the same Point/Tone grid as its IQ.
    axes = (
        VariableAxisSpec.generated_arange("Point", "", dtype=np.int64),
        VariableAxisSpec("Tone", "", field_name="tone_indices", dtype=np.int64),
    )

    def build(data: GroupedLoadData[T]) -> MuxSweepResult:
        # Reject mismatched saved grids before rebuilding the measurement arrays.
        coordinates = data.variable("coordinates")
        signals = data.variable("signals")
        if any(
            not np.array_equal(a, b)
            for a, b in zip(coordinates.axes, signals.axes, strict=True)
        ):
            raise ValueError("Mux coordinate and signal axes must match")
        expected = np.arange(coordinates.z.shape[1], dtype=np.int64)
        if not np.array_equal(coordinates.axes[0], expected):
            raise ValueError("Mux Point axis must preserve original zip indices")

        result = MuxSweepResult(
            coordinates.axes[1].astype(np.int64),
            coordinates.z.astype(np.float64),
            signals.z.astype(np.complex128),
        )
        validate_mux_result(result)
        return result

    # Declare unit conversion without reading or writing a measurement file.
    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                variable="coordinates",
                axes=axes,
                z=VariableZSpec(
                    field_name="coordinates",
                    label=coordinate_label,
                    unit=unit,
                    scale=scale,
                    dtype=np.float64,
                ),
            ),
            VariableSpec(
                variable="signals",
                axes=axes,
                z=VariableZSpec(
                    field_name="signals",
                    label="Signal",
                    unit="a.u.",
                    dtype=np.complex128,
                ),
            ),
        ),
        result_type=MuxSweepResult,
        cfg_type=cfg_type,
        tag=tag,
        result_builder=build,
        result_validator=validate_mux_result,
        cfg_schema_version="1.0",
    )
