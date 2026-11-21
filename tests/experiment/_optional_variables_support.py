"""Synthetic sweep with an optional diagnostic on a separate grid."""

from dataclasses import dataclass

import numpy as np
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    RunRecord,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from tests._native_support import native_metadata


class DiagnosticCfg(ConfigBase):
    name: str = "optional diagnostics"


@dataclass(frozen=True)
class DiagnosticResult:
    points: np.ndarray
    signal: np.ndarray
    diagnostics: np.ndarray | None = None


def build_diagnostic(data: GroupedLoadData[DiagnosticCfg]) -> DiagnosticResult:
    signal = data.variable("data")
    diagnostic = data.variables.get(DataVariable("diagnostics"))
    return DiagnosticResult(
        signal.axes[0], signal.z, diagnostic.z if diagnostic is not None else None
    )


def diagnostic_spec() -> GroupedAxesSpec[DiagnosticResult, DiagnosticCfg]:
    points = VariableAxisSpec(label="Time", unit="s", field_name="points", scale=1e-6)
    diagnostics = VariableSpec(
        "diagnostics",
        (VariableAxisSpec(label="Statistic", unit="", values=(0, 1)), points),
        VariableZSpec("diagnostics", label="Diagnostics", unit="a.u."),
        required=False,
    )
    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (points,),
                VariableZSpec("signal", label="Signal", unit="V", dtype=np.complex128),
            ),
            diagnostics,
        ),
        result_type=DiagnosticResult,
        cfg_type=DiagnosticCfg,
        tag="test/optional-diagnostics",
        result_builder=build_diagnostic,
    )


def diagnostic_record(*, present: bool) -> RunRecord[DiagnosticCfg, DiagnosticResult]:
    return RunRecord(
        cfg=DiagnosticCfg(),
        result=DiagnosticResult(
            np.array([1.0, 2.0]),
            np.array([1 + 2j, 3 - 4j]),
            np.array([[2.0, 10.0], [3.0, 16.0]]) if present else None,
        ),
        metadata=native_metadata("test/optional-diagnostics").model_dump(mode="json"),
    )
