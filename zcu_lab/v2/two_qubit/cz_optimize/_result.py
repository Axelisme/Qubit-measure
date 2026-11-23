"""Native results and the named physical CZ parameter columns."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.core.waveform_format import WaveformSamples
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment.axes_spec import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from ._shape import KNOTS, StartStyle, render_candidate


class _Parameter(NamedTuple):
    name: str
    label: str
    unit: str


PARAMETERS = (
    *(_Parameter(f"i_knot_{i}", f"I knot {i}", "") for i in range(1, KNOTS + 1)),
    *(_Parameter(f"q_knot_{i}", f"Q knot {i}", "") for i in range(1, KNOTS + 1)),
    _Parameter("gain_scale", "Field multiplier g", ""),
    _Parameter("local_z_qubit1", "Qubit1 local Z", "rad"),
    _Parameter("local_z_qubit2", "Qubit2 local Z", "rad"),
)


@dataclass(frozen=True)
class CZOptimizeResult:
    """Measured counts and physical parameters, including the calibrated start.

    seeds is int64 (seed,). sums is float64 (evaluation, seed, 3, 3), in
    Qubit1/Qubit2 order with classes g, e, other. params is float64 (evaluation,
    parameter) in PARAMETERS order (I knots, Q knots, physical gain g,
    radians for Qubit1 and Qubit2). Row zero holds the CMA initial mean; the
    measured evaluation zero plays the original pulse instead. peak is float64
    (evaluation,), the envelope magnitude which gates only CMA evaluations.
    Unevaluated fields and unmeasured count tables are NaN.

    start_iq is the complex128 display envelope of evaluation zero in Qubit1
    DDS gain units. start_style identifies the original waveform. Duration
    and cycle period are seconds; samples_per_cycle describes the display
    and CMA grid. carrier_hz is the shared carrier. arb_gains holds raw gains
    at physical g=1 for both lines; field_at_one gives Qubit1 DDS gain at g=1.
    start_asset_fingerprints is the optional UTF-8 map from starting paths to
    their Run-time SHA-256, retained when cfg cannot be restored.
    """

    seeds: NDArray[np.int64]
    params: NDArray[np.float64]
    peak: NDArray[np.float64]
    sums: NDArray[np.float64]
    start_iq: NDArray[np.complex128]
    start_style: StartStyle
    start_duration_s: float
    cycle_period_s: float
    samples_per_cycle: int
    carrier_hz: float
    arb_gains: NDArray[np.float64]
    field_at_one: float
    start_asset_fingerprints: NDArray[np.uint8] | None = None

    def render_candidate(self, index: int) -> WaveformSamples:
        """Render the CMA sample grid at a finite parameter row (zero is the fit).

        Returns dimensionless I/Q and times in µs including the end boundary.
        Does not recover the actual starting pulse played at evaluation zero.
        An invalid row index or non-finite row raises.
        """
        params = self.params[index]
        if not np.all(np.isfinite(params)):
            raise ValueError(f"Evaluation {index} has no complete parameter row")
        return render_candidate(
            params,
            duration_us=self.start_duration_s * 1e6,
            cycle_us=self.cycle_period_s * 1e6,
            samples_per_cycle=self.samples_per_cycle,
        )


_EVAL = VariableAxisSpec.generated_arange(label="Evaluation", unit="")
_START_ASSETS = DataVariable("start_asset_fingerprints")


def _scalar(
    name: str, label: str, unit: str, *, dtype: type = np.float64
) -> VariableSpec:
    return VariableSpec(
        name,
        (),
        VariableZSpec(name, label=label, unit=unit, dtype=dtype),
        required=False,
    )


def _restore[CfgT: ConfigBase](data: GroupedLoadData[CfgT]) -> CZOptimizeResult:
    # ADR-0069: reject the old result layout instead of guessing parameter
    # units or manufacturing display samples from an earlier parameterization.
    if PARAMETERS[0].name not in data.variables:
        raise ValueError(
            "CZ optimization result format has changed to fixed-duration arb. "
            "Open an older run with the version that saved it."
        )
    sums = data.variable("sums")
    fingerprints = data.variables.get(_START_ASSETS)
    return CZOptimizeResult(
        seeds=sums.axes[2],
        params=np.column_stack([data.variable(param.name).z for param in PARAMETERS]),
        peak=data.variable("peak").z,
        sums=sums.z,
        start_iq=data.variable("start_iq").z,
        start_style=StartStyle(data.variable("start_style").z.item()),
        start_duration_s=float(data.variable("start_duration_s").z.item()),
        cycle_period_s=float(data.variable("cycle_period_s").z.item()),
        samples_per_cycle=int(data.variable("samples_per_cycle").z.item()),
        carrier_hz=float(data.variable("carrier_hz").z.item()),
        arb_gains=data.variable("arb_gains").z,
        field_at_one=float(data.variable("field_at_one").z.item()),
        start_asset_fingerprints=fingerprints.z if fingerprints is not None else None,
    )


def axes_spec[CfgT: ConfigBase](
    cfg_type: type[CfgT],
) -> GroupedAxesSpec[CZOptimizeResult, CfgT]:
    """Declare CZ result groups for native roundtrip and named Labber columns."""
    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                "sums",
                (
                    VariableAxisSpec(
                        label="Qubit2 class", unit="", values=(0, 1, 2), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Qubit1 class", unit="", values=(0, 1, 2), dtype=np.int64
                    ),
                    VariableAxisSpec(
                        label="Seed", unit="", field_name="seeds", dtype=np.int64
                    ),
                    _EVAL,
                ),
                VariableZSpec("sums", label="Joint class counts", unit=""),
            ),
            VariableSpec(
                "peak", (_EVAL,), VariableZSpec("peak", label="Envelope peak", unit="")
            ),
            # The builder requires these columns. Declaring them optional to
            # the generic reader lets it report the experiment's format change.
            *(
                VariableSpec(
                    param.name,
                    (_EVAL,),
                    VariableZSpec(
                        "params", label=param.label, unit=param.unit, index=i
                    ),
                    required=False,
                )
                for i, param in enumerate(PARAMETERS)
            ),
            VariableSpec(
                "start_iq",
                (VariableAxisSpec.generated_arange(label="DAC sample", unit=""),),
                VariableZSpec(
                    "start_iq", label="Start display I/Q", unit="", dtype=np.complex128
                ),
                required=False,
            ),
            VariableSpec(
                _START_ASSETS,
                (VariableAxisSpec.generated_arange(label="Byte", unit=""),),
                VariableZSpec(
                    "start_asset_fingerprints",
                    label="Starting CZ asset fingerprints",
                    unit="",
                    dtype=np.uint8,
                ),
                required=False,
            ),
            _scalar("start_style", "Start style ID", "", dtype=np.int64),
            _scalar("start_duration_s", "Start duration", "s"),
            _scalar("cycle_period_s", "Generator fabric period", "s"),
            _scalar(
                "samples_per_cycle", "DAC samples per fabric cycle", "", dtype=np.int64
            ),
            _scalar("carrier_hz", "CZ drive carrier", "Hz"),
            _scalar("field_at_one", "Qubit1 DDS gain at g=1", ""),
            VariableSpec(
                "arb_gains",
                (
                    VariableAxisSpec(
                        label="Qubit position", unit="", values=(1, 2), dtype=np.int64
                    ),
                ),
                VariableZSpec("arb_gains", label="Raw arb gains at g=1", unit=""),
                required=False,
            ),
        ),
        result_type=CZOptimizeResult,
        cfg_type=cfg_type,
        tag="two_qubit/cz_optimize",
        result_builder=_restore,
        cfg_schema_version="4.0",
        labber_layout="independent",
        labber_variables=(*(param.name for param in PARAMETERS), "peak"),
    )
