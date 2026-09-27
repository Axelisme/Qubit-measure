from __future__ import annotations

from typing import Any, NotRequired, cast

from typing_extensions import TypedDict

from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.resources.qubit_params import QubitParams


class FluxDepFitResult(TypedDict):
    params: dict[str, float]
    flux_half: float
    flux_int: float
    flux_period: float
    plot_transitions: TransitionDict
    timestamp: NotRequired[str]


class DispersiveResult(TypedDict):
    bare_rf: float
    g: float
    timestamp: NotRequired[str]


class ProjectIdentity(TypedDict):
    chip_name: str
    qubit_name: str
    resonator_name: NotRequired[str]


class ResultData(TypedDict):
    name: str
    schema_version: NotRequired[int]
    project: NotRequired[ProjectIdentity]
    fluxdep_fit: NotRequired[FluxDepFitResult]
    dispersive: NotRequired[DispersiveResult]


def _project_from_name(name: str) -> ProjectIdentity | None:
    parts = [part.strip() for part in name.split("/")]
    if len(parts) == 2 and all(parts):
        return {
            "chip_name": parts[0],
            "qubit_name": parts[1],
            "resonator_name": "unknown",
        }
    return None


def dump_result(
    path: str,
    name: str,
    fluxdep_fit: FluxDepFitResult | None = None,
    dispersive: DispersiveResult | None = None,
    schema_version: int | None = None,
    project: ProjectIdentity | None = None,
) -> None:
    result: dict[str, Any] = {"name": name}
    project_identity = project or _project_from_name(name)
    if project_identity is not None:
        result["schema_version"] = schema_version or 1
        result["project"] = project_identity
    elif schema_version is not None:
        result["schema_version"] = schema_version
    if fluxdep_fit is not None:
        result["fluxdep_fit"] = fluxdep_fit
    if dispersive is not None:
        result["dispersive"] = dispersive

    QubitParams(path).replace_raw(result)


def load_result(path: str) -> ResultData:
    """Load the result from a json file"""

    return cast(ResultData, QubitParams(path, readonly=True).to_raw())


def update_result(path: str, update_dict: dict[str, Any]) -> None:
    QubitParams(path).update_raw(update_dict)
