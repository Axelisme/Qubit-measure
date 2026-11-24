"""Project the four historical coherence Labber layouts to current Native data."""

from dataclasses import dataclass, replace

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import Container, Variable
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    GroupedAxesSpec,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)
from zcu_tools.experiment.axes_spec import LoadedVariableData
from zcu_tools.experiment.run_values import encode_run_values

from zcu_lab.v2.support.mux_coherence import (
    MuxCoherenceResult,
    validate_coherence_result,
)

_TONE = VariableAxisSpec(
    label="Tone", unit="", field_name="tone_indices", dtype=np.int64
)


def _load_tone_log(name: str, label: str, variable: Variable) -> LoadedVariableData:
    return VariableSpec(
        name,
        (_TONE,),
        VariableZSpec(
            field_name=name, label=label, unit="Hz", scale=MHZ_TO_HZ, dtype=np.float64
        ),
    ).loaded_from_variable(variable, context=name)


def project_mux_coherence[Cfg: ConfigBase](
    payload: Container, *, spec: GroupedAxesSpec[MuxCoherenceResult, Cfg]
) -> Container:
    """Project a validated historical three-log mux payload using the leaf spec.

    Typed Tone positions must match in original order across the logs. Preserve
    IQ, disk coordinates and the raw Labber group without rewriting them. Carry
    the applied detune into the Native document and discard the redundant carrier
    from Native vars. Raise ValueError for unaligned or invalid source values.
    No I/O occurs.
    """
    signals = payload.vars["signals"]
    signal_spec = next(
        item for item in spec.variables if item.data_variable == "signals"
    )
    iq = signal_spec.loaded_from_variable(signals, context="signals")
    tone_indices = iq.axes[1]
    detunes = _load_tone_log(
        "actual_detunes", "Actual detune", payload.vars["actual_detunes"]
    )
    carrier = _load_tone_log(
        "source_frequencies", "Source frequency", payload.vars["source_frequencies"]
    )
    for log in (detunes, carrier):
        if not np.array_equal(log.axes[0], tone_indices):
            raise ValueError(
                f"{log.variable}/Tone: must match signals/Tone in saved order"
            )
    if not np.isfinite(carrier.z).all():
        raise ValueError("source_frequencies: historical carrier log must be finite")
    result = MuxCoherenceResult(tone_indices, iq.axes[0], iq.z, detunes.z)
    validate_coherence_result(result)
    variables = {"signals": signals}
    run_values = encode_run_values(
        variables,
        result=result,
        declarations=spec.run_values,
        version=spec.run_values_version,
    )
    return replace(
        payload, vars=variables, docs={**payload.docs, "run_values": run_values}
    )


@dataclass(frozen=True)
class _UnknownDetune:
    true_activate_detune: None = None


def project_unknown_detune[Result, Cfg: ConfigBase](
    payload: Container, *, spec: GroupedAxesSpec[Result, Cfg]
) -> Container:
    """Encode the absent detune for a historical single-qubit coherence log.

    payload supplies historical data/Time and the opaque Labber group. The
    current Echo or Ramsey spec declares a nullable scalar detune, which is
    written as unknown. Preserve all payload data and documents without I/O.
    """
    run_values = encode_run_values(
        payload.vars,
        result=_UnknownDetune(),
        declarations=spec.run_values,
        version=spec.run_values_version,
    )
    return replace(payload, docs={**payload.docs, "run_values": run_values})
