"""Author-facing generator recipe contracts; GUI policy stays in the GUI owner.

Recipes receive a RecipeSession from the driver, not a transport or cfg publication.
Run, analysis and writeback questions are yielded once. The driver delivers their
status or throws a failure back into that yield expression. Session close closes
the generator, so ordinary Python finally blocks still run.

Cfg preparation uses one fixed connection and the GUI's returned publications.
The generator operation/driver integration is still being prepared; this module
is not yet wired into tool assembly.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, NotRequired, TypedDict, TypeGuard

from zcu_tools.mcp.measure.analysis_execution import (
    AnalysisWriteback,
    ExecutionSnapshot,
)
from zcu_tools.mcp.measure.execution_reply import SummaryEstimate, SummaryParameter
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.tool_context import MeasureToolContext

RecipeScalar = str | int | float | bool | None
RecipeParameter = RecipeScalar | list[RecipeScalar]
StepStatus = Literal["completed", "finished_early", "cancelled"]
WritebackDecision = Literal["accepted", "skipped"]
AnalysisStage = Literal["primary", "post"]


class RecipePropertySchema(TypedDict):
    """Handwritten JSON Schema for an existing scalar or array recipe parameter.

    type lists the accepted JSON types, including null when optional. minLength
    constrains strings; items describes array elements; minItems constrains the
    array length. No annotations-to-schema generation takes place.
    """

    type: str | list[str]
    minLength: NotRequired[int]
    items: NotRequired[RecipePropertySchema]
    minItems: NotRequired[int]


class RecipeInputSchema(TypedDict):
    """An object schema whose properties name the recipe's typed keyword inputs.

    additionalProperties rejects undeclared keys when false. required names the
    mandatory keys; omitted keys otherwise use the recipe's Python defaults.
    """

    type: Literal["object"]
    additionalProperties: bool
    properties: dict[str, RecipePropertySchema]
    required: NotRequired[list[str]]


@dataclass(frozen=True)
class MissingParameter:
    """One missing tool parameter and the instruction for supplying it.

    parameter is the public keyword name; reason explains the absent calibration
    or input. Both strings must be non-empty.
    """

    parameter: str
    reason: str


class RecipeNeedsParameters(Exception):
    """End a recipe with a needs_parameters receipt instead of a GUI failure.

    missing must be a non-empty tuple of MissingParameter values with non-empty
    strings. The same tuple is exposed as missing. Invalid declarations raise
    ValueError; the exception does not close a generator or perform a GUI action.
    """

    def __init__(self, missing: tuple[MissingParameter, ...]) -> None:
        if not missing or any(
            not item.parameter or not item.reason for item in missing
        ):
            raise ValueError(
                "missing parameters must contain non-empty names and reasons"
            )
        self.missing = missing
        super().__init__(
            "; ".join(f"{item.parameter}: {item.reason}" for item in missing)
        )


@dataclass(frozen=True)
class RawSaveReceipt:
    """Capture of an admitted raw save, not a promise about a reserved path.

    status is not_started, saving, saved, failed or unknown. reserved_path is the
    planned file path; path is a confirmed saved path. None means not captured.
    operation_outcome retains the native terminal reply, or None before capture.
    Native fields are opaque diagnostic facts, not a second operation policy.
    """

    status: Literal["not_started", "saving", "saved", "failed", "unknown"] = (
        "not_started"
    )
    reserved_path: str | None = None
    path: str | None = None
    operation_outcome: Mapping[str, object] | None = None


class WritebackItemReceipt(TypedDict):
    """One GUI-confirmed write, retaining its native before/after values.

    id is the current pane's opaque candidate ID; kind is its GUI-owned writeback
    kind; target is the native destination description. before and after are
    opaque native values and may be None. An unconfirmed write is never listed.
    """

    id: str
    kind: str
    target: object
    before: object
    after: object


class WritebackStageReceipt(TypedDict):
    """Confirmed writes for the named primary or post analysis stage."""

    stage: AnalysisStage
    written: list[WritebackItemReceipt]


class WritebackFailure(TypedDict):
    """First failed writeback step: GUI code/reason may be None; message is text."""

    code: str | None
    reason: str | None
    message: str


class WritebackReceipt(TypedDict):
    """Progress of a Primary-then-Post writeback without rollback or retry.

    tab is the GUI locator. status is finished or failed. completed contains only
    confirmed writes grouped by stage. skipped names stages with no selected
    candidates; not_started names stages not attempted after the first failure.
    Failed replies additionally name failed_stage and error. The partial-writes
    flag says only that a write RPC was attempted in the failed stage, not which
    unconfirmed items were written. Finished replies omit all three failure keys.
    """

    tab: str
    status: Literal["finished", "failed"]
    completed: list[WritebackStageReceipt]
    skipped: list[AnalysisStage]
    not_started: list[AnalysisStage]
    failed_stage: NotRequired[AnalysisStage]
    error: NotRequired[WritebackFailure]
    failed_stage_may_have_partial_writes: NotRequired[bool]


class WritebackError(RuntimeError):
    """Report failed tab.accept while retaining its confirmed progress receipt.

    receipt must have status failed. The same receipt is exposed as receipt.
    ValueError rejects a finished receipt. No retry or rollback is performed.
    """

    def __init__(self, receipt: WritebackReceipt) -> None:
        if receipt["status"] != "failed":
            raise ValueError("WritebackError requires a failed receipt")
        self.receipt = receipt
        error = receipt.get("error")
        super().__init__(error["message"] if error else "writeback failed")


@dataclass(frozen=True)
class RecipeWritebackPreview:
    """Captured proposals for a question; no live query or authorization.

    primary and post are their completed analysis captures, including destination
    context. None means that stage has no captured preview. Selection resolves
    the stable target_name across both stages before asking the question.
    """

    primary: AnalysisWriteback | None = None
    post: AnalysisWriteback | None = None


@dataclass(frozen=True)
class RecipeAnalysis:
    """Completed analysis capture delivered by an AnalyzeOperation yield.

    snapshot contains the stage's native result, actual parameters, invalid
    fields, artifacts, preview and writeback facts. No method re-reads candidates.
    """

    snapshot: ExecutionSnapshot


class RunOperation:
    """Opaque, non-iterable Run handle created by RecipeTab.run.

    Yield once to receive (RecipeRun, completed/finished_early/cancelled). A
    between-yield cancel instead delivers (None, cancelled) without sending Run.
    Failure or superseded source is thrown back at the yield expression.
    """


class AnalyzeOperation:
    """Opaque, non-iterable analysis handle created by RecipeRun.analyze.

    Yield once to receive (RecipeAnalysis, completed) or (None, cancelled).
    Interactive handoff does not finish the yield; GUI done resumes observation.
    Failure or superseded source is thrown back at the yield expression.
    """


class WritebackQuestion:
    """Opaque, non-iterable captured question created by propose_writeback.

    Yield once to pause until answer delivers (accepted/skipped, completed), or
    cancellation delivers (None, cancelled). Answer itself never writes data.
    """


class _CfgEdit(TypedDict):
    """One wire edit: path is a public field path; value is GUI input data."""

    path: list[str]
    value: object


@dataclass
class _TabBinding:
    """Cfg preparation state hidden behind an author tab handle.

    tools is the execution's fixed connection. tab is the GUI locator.
    publication is the last returned cfg observation, never refreshed on failure.
    calibrations contains opaque md values from the declared context observation.
    libraries contains the observed module-library names. origins records sources
    by public/derived field name for the later Run capture; flux_unit is its asserted unit.
    """

    tools: MeasureToolContext
    tab: str
    publication: dict[str, object]
    calibrations: dict[str, object]
    libraries: frozenset[str]
    origins: dict[str, str] = field(default_factory=dict)
    flux_unit: str | None = None


def _cfg_object(value: object) -> dict[str, object]:
    """Decode an object at the GUI wire boundary without accepting bad keys."""
    if not isinstance(value, dict):
        raise GuiRpcError("Expected a GUI object", reason="incompatible_wire")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise GuiRpcError("Expected GUI string keys", reason="incompatible_wire")
        result[key] = item
    return result


def _field_path(name: str) -> list[str]:
    if not name or any(not part or part != part.strip() for part in name.split(".")):
        raise ValueError("field must be a non-empty dotted cfg field name")
    return name.split(".")


def _calibration_key(calibration: Literal["resonator", "qubit"]) -> str:
    if calibration == "resonator":
        return "r_f"
    if calibration == "qubit":
        return "q_f"
    raise ValueError("calibration must be resonator or qubit")


def _check_points(expts: object) -> None:
    if expts is not None and (
        not isinstance(expts, int) or isinstance(expts, bool) or expts < 2
    ):
        raise ValueError("expts must be an integer of at least 2")


def _finite_frequency(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


class RecipeSession:
    """One execution's fixed GUI binding, created and passed by the driver.

    tools is the owning session's tool context. Construction captures its current
    GUI connection (or performs the initial attach); subsequent helpers never
    reconnect, retry stale requests or synthesize GUI defaults. Fast edits remain
    available after cancellation; the recipe decides its policy.
    """

    def __init__(self, tools: MeasureToolContext) -> None:
        self._tools = tools.bound()

    def open_tab(self, adapter: str, *, reuse: str | None = None) -> RecipeTab:
        """Open adapter's tab, or validate/reset the named reusable tab.

        adapter is a non-empty public GUI adapter ID. reuse=None creates a fresh
        tab; otherwise reuse is a non-empty GUI locator. Busy, mismatched adapters
        and invalid locators fail without creating a fallback. Observe context
        sources once and capture the cfg ref for edits and Run. GUI/connection
        errors propagate; invalid names raise ValueError before any GUI request.
        """
        if not adapter or not adapter.strip():
            raise ValueError("adapter must be a non-empty GUI adapter ID")
        if reuse is not None and (not reuse or not reuse.strip()):
            raise ValueError("reuse must be a non-empty GUI tab locator")
        sources = self._tools.send_gui_rpc("context.snapshot", {})
        calibrations = _cfg_object(sources["md"])
        libraries = _cfg_object(_cfg_object(sources["ml"])["modules"])
        if reuse is None:
            created = self._tools.send_gui_rpc("tab.new", {"adapter_name": adapter})
            tab = created["tab_id"]
            if not isinstance(tab, str) or not tab:
                raise GuiRpcError(
                    "Invalid tab creation reply", reason="incompatible_wire"
                )
            publication = self._tools.send_gui_rpc("tab.get_cfg", {"tab_id": tab})
        else:
            tab = reuse
            observed = self._tools.send_gui_rpc("tab.snapshot", {"tab_id": tab})["tabs"]
            if not isinstance(observed, list) or len(observed) != 1:
                raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
            snapshot = _cfg_object(observed[0])
            if snapshot["tab_id"] != tab:
                raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
            if snapshot["adapter_name"] != adapter:
                raise GuiRpcError(
                    "Tab has a different experiment", reason="wrong_experiment"
                )
            interaction = _cfg_object(snapshot["interaction"])
            if any(
                interaction[key]
                for key in ("is_running", "is_analyzing", "is_saving_data")
            ):
                raise GuiRpcError("Tab is busy", reason="tab_busy")
            observed_cfg = self._tools.send_gui_rpc("tab.get_cfg", {"tab_id": tab})
            publication = self._tools.send_gui_rpc(
                "tab.reset_cfg", {"tab_id": tab, "expected": observed_cfg["cfg_ref"]}
            )
        return RecipeTab(
            _TabBinding(
                self._tools,
                tab,
                _cfg_object(publication),
                calibrations,
                frozenset(libraries),
            )
        )


class RecipeTab:
    """Opaque tab handle returned by RecipeSession.open_tab.

    field names are dot-separated public cfg fields (for example rounds or
    modules.readout), not wire path lists. Methods preserve the observed cfg ref
    and record parameter sources. GUI errors propagate without re-read, retry or
    rollback; required calibration gaps raise RecipeNeedsParameters. Authors obtain
    this handle from open_tab rather than constructing the private binding.
    """

    def __init__(self, binding: _TabBinding) -> None:
        self._binding = binding

    def _node(self, name: str) -> dict[str, object]:
        node = _cfg_object(self._binding.publication["tree"])
        for part in _field_path(name):
            children = _cfg_object(node.get("children", {}))
            if part not in children:
                raise ValueError(f"Unknown cfg field: {name}")
            node = _cfg_object(children[part])
        return node

    def _edit(self, edits: list[_CfgEdit]) -> None:
        if edits:
            reply = self._binding.tools.send_gui_rpc(
                "tab.edit_cfg",
                {
                    "tab_id": self._binding.tab,
                    "expected": self._binding.publication["cfg_ref"],
                    "edits": edits,
                },
            )
            self._binding.publication = _cfg_object(reply)

    def _reference(self, field: str) -> dict[str, object]:
        node = self._node(field)
        if node["kind"] != "reference":
            raise ValueError(f"Not a module reference field: {field}")
        return node

    def use_library(
        self, field: str, name: str | None, *, required: str | None = None
    ) -> None:
        """Select name for a module field; None retains its current GUI value.

        required names the missing tool parameter if no usable reference exists.
        Unknown or invalid explicit library references fail rather than fall back.
        """
        node = self._reference(field)
        if name is not None:
            if not name or not name.strip():
                raise ValueError("library name must be non-empty")
            self._edit([{"path": _field_path(field), "value": {"__ref": name}}])
            node = self._reference(field)
            if node.get("error") or not node.get("valid") or node.get("ref") != name:
                raise GuiRpcError(f"Invalid {field} reference", reason="invalid_cfg")
        elif required is not None and (
            not isinstance(node.get("ref"), str)
            or node.get("ref") not in self._binding.libraries
            or not node.get("valid")
            or node.get("error")
        ):
            raise RecipeNeedsParameters(
                (
                    MissingParameter(
                        required, f"Provide a valid {field} library reference"
                    ),
                )
            )
        self._binding.origins[field] = "explicit" if name is not None else "gui_default"

    def disable_library(self, field: str) -> None:
        """Disable an optional module field explicitly; reject unknown fields."""
        self._reference(field)
        self._edit([{"path": _field_path(field), "value": {"__ref": None}}])
        node = self._reference(field)
        if node.get("ref") is not None or node.get("error"):
            raise GuiRpcError(f"Cannot disable {field}", reason="invalid_cfg")
        self._binding.origins[field] = "disabled"

    def set(self, field: str, value: RecipeParameter | None) -> None:
        """Set a scalar cfg field; None keeps its GUI default and records that source.

        Reject unknown/non-scalar fields locally. The GUI validates supplied values;
        a returned field error raises GuiRpcError without retrying the edit.
        """
        node = self._node(field)
        if node["kind"] != "scalar":
            raise ValueError(f"Not a scalar cfg field: {field}")
        if value is not None:
            self._edit([{"path": _field_path(field), "value": value}])
            node = self._node(field)
            state = _cfg_object(node["input"])
            if (
                not node.get("valid")
                or state.get("error")
                or state.get("validation_error")
            ):
                raise GuiRpcError(f"Invalid {field} value", reason="invalid_cfg")
        self._binding.origins[field] = (
            "explicit" if value is not None else "gui_default"
        )

    def _frequency_fields(self, field: str) -> tuple[list[str], str | None]:
        """Resolve scalar/readout-root fields and the enclosing library reference."""
        node = self._node(field)
        if node["kind"] == "reference":
            children = _cfg_object(node["children"])
            if "ro_freq" in children:
                paths = [f"{field}.ro_freq"]
            elif "pulse_cfg" in children and "ro_cfg" in children:
                paths = [f"{field}.pulse_cfg.freq", f"{field}.ro_cfg.ro_freq"]
            else:
                raise GuiRpcError("Unsupported readout shape", reason="invalid_cfg")
            reference = node.get("ref")
            return paths, reference if isinstance(reference, str) else None
        if node["kind"] != "scalar":
            raise ValueError(f"Not a frequency field: {field}")
        ancestors = _field_path(field)[:-1]
        while ancestors:
            parent = self._node(".".join(ancestors))
            if parent["kind"] == "reference":
                reference = parent.get("ref")
                return [field], reference if isinstance(reference, str) else None
            ancestors.pop()
        return [field], None

    def _usable_frequency(self, field: str) -> bool:
        node = self._node(field)
        state = _cfg_object(node["input"])
        return bool(
            node.get("valid")
            and not state.get("error")
            and not state.get("validation_error")
            and _finite_frequency(state.get("resolved"))
        )

    def set_frequency(
        self,
        field: str,
        frequency_mhz: float | None = None,
        *,
        calibration: Literal["resonator", "qubit"],
        prefer_library: bool = True,
        required: str,
    ) -> None:
        """Set a frequency scalar or readout module root in MHz.

        Choose explicit finite frequency_mhz, then usable current library when
        prefer_library, then the named calibration. If none exists, report required
        as a missing tool parameter. Handle pulse and single-frequency readout cfg
        in this helper; callers do not inspect the GUI's nested representation.
        """
        key = _calibration_key(calibration)
        if frequency_mhz is not None and not _finite_frequency(frequency_mhz):
            raise ValueError("frequency_mhz must be finite and not boolean")
        paths, reference = self._frequency_fields(field)
        calibrated = self._binding.calibrations.get(key)
        edits: list[_CfgEdit] = []
        origins: dict[str, str] = {}
        for path in paths:
            if frequency_mhz is not None:
                edits.append({"path": _field_path(path), "value": frequency_mhz})
                origins[path] = "explicit"
            elif (
                prefer_library
                and reference in self._binding.libraries
                and self._usable_frequency(path)
            ):
                origins[path] = f"library:{reference}"
            elif _finite_frequency(calibrated):
                edits.append({"path": _field_path(path), "value": {"__expr": key}})
                origins[path] = key
            else:
                raise RecipeNeedsParameters(
                    (
                        MissingParameter(
                            required,
                            f"Provide an explicit frequency, valid library, or calibrated {key}",
                        ),
                    )
                )
        self._edit(edits)
        for path in paths:
            if not self._usable_frequency(path):
                raise GuiRpcError(f"Invalid {path} frequency", reason="invalid_cfg")
        self._binding.origins.update(origins)

    def _sweep_inputs(self, field: str) -> dict[str, object]:
        node = self._node(field)
        if node["kind"] != "sweep":
            raise ValueError(f"Not a sweep cfg field: {field}")
        return _cfg_object(node["inputs"])

    def _write_sweep(self, field: str, values: dict[str, object]) -> None:
        self._sweep_inputs(field)
        if not values:
            return
        self._edit([{"path": _field_path(field), "value": values}])
        inputs = self._sweep_inputs(field)
        if not self._node(field).get("valid") or any(
            _cfg_object(inputs[key]).get("error")
            or _cfg_object(inputs[key]).get("validation_error")
            for key in values
        ):
            raise GuiRpcError(f"Invalid {field} sweep", reason="invalid_cfg")

    def _linewidth_span(
        self, field: str, center_key: str, width_key: str
    ) -> str | None:
        """Preserve the GUI's width expression instead of choosing a new span."""
        width = self._binding.calibrations.get(width_key)
        inputs = self._sweep_inputs(field)
        edges = [_cfg_object(inputs[key]) for key in ("start", "stop")]
        if not _finite_frequency(width) or width <= 0:
            return None
        expressions: list[str] = []
        for edge in edges:
            raw = edge["raw"]
            if (
                edge["mode"] != "expression"
                or not isinstance(raw, str)
                or not re.search(rf"\b{width_key}\b", raw)
            ):
                return None
            expressions.append(re.sub(rf"\b{center_key}\b", "0", raw))
        return f"({expressions[1]}) - ({expressions[0]})"

    def set_frequency_sweep(
        self,
        field: str,
        *,
        calibration: Literal["resonator", "qubit"],
        center_mhz: float | None = None,
        span_mhz: float | None = None,
        expts: int | None = None,
    ) -> None:
        """Set MHz sweep endpoints/points, preserving omitted GUI-derived inputs.

        Missing center uses the named frequency calibration. Missing span retains
        its calibrated linewidth expression; a supplied span must be positive.
        expts is an integer of at least 2; None retains the GUI count. Missing
        calibration/linewidth raises RecipeNeedsParameters. Invalid explicit
        inputs raise ValueError; invalid resolved ranges raise GuiRpcError.
        """
        key = _calibration_key(calibration)
        width_key = "rf_w" if calibration == "resonator" else "qf_w"
        _check_points(expts)
        if center_mhz is not None and not _finite_frequency(center_mhz):
            raise ValueError("center_mhz must be finite and not boolean")
        if span_mhz is not None and (not _finite_frequency(span_mhz) or span_mhz <= 0):
            raise ValueError("span_mhz must be finite and positive")
        self._sweep_inputs(field)
        missing: list[MissingParameter] = []
        center: str | float = key if center_mhz is None else center_mhz
        if center_mhz is None and not _finite_frequency(
            self._binding.calibrations.get(key)
        ):
            missing.append(
                MissingParameter("center_mhz", f"No finite {key} calibration")
            )
        span: str | float | None = span_mhz
        if span_mhz is None:
            span = self._linewidth_span(field, key, width_key)
            if span is None:
                missing.append(
                    MissingParameter("span_mhz", "No GUI linewidth-derived range")
                )
        if missing:
            raise RecipeNeedsParameters(tuple(missing))
        values: dict[str, object] = {
            "start": {"__expr": f"({center}) - ({span}) / 2"},
            "stop": {"__expr": f"({center}) + ({span}) / 2"},
        }
        if expts is not None:
            values["expts"] = expts
        self._write_sweep(field, values)
        inputs = self._sweep_inputs(field)
        start = _cfg_object(inputs["start"])["resolved"]
        stop = _cfg_object(inputs["stop"])["resolved"]
        if (
            not _finite_frequency(start)
            or not _finite_frequency(stop)
            or stop <= start
            or not _finite_frequency(stop - start)
        ):
            raise GuiRpcError("Invalid resolved frequency range", reason="invalid_cfg")
        self._binding.origins[field] = (
            "gui_calibration" if center_mhz is None and span_mhz is None else "explicit"
        )
        self._binding.origins["center_mhz"] = key if center_mhz is None else "explicit"
        self._binding.origins["span_mhz"] = (
            "gui_linewidth" if span_mhz is None else "explicit"
        )
        self._binding.origins[f"{field}.expts"] = (
            "gui_default" if expts is None else "explicit"
        )

    def set_flux_sweep(
        self,
        field: str,
        *,
        start: float | None = None,
        stop: float | None = None,
        expts: int | None = None,
    ) -> None:
        """Set flux sweep coordinates without converting their device unit.

        start/stop must both be finite, distinct or both omitted. Omitted endpoints
        retain valid flx_half/flx_int expressions; absent calibration raises
        RecipeNeedsParameters. expts is an integer of at least 2; None retains
        the GUI count. Invalid explicit inputs raise ValueError; invalid resolved
        ranges raise GuiRpcError.
        """
        _check_points(expts)
        if (start is None) != (stop is None) or (
            start is not None
            and stop is not None
            and (
                not _finite_frequency(start)
                or not _finite_frequency(stop)
                or start == stop
            )
        ):
            raise ValueError(
                "flux endpoints must both be finite and distinct, or both omitted"
            )
        inputs = self._sweep_inputs(field)
        if start is None:
            half = self._binding.calibrations.get("flx_half")
            integer = self._binding.calibrations.get("flx_int")
            if (
                not _finite_frequency(half)
                or not _finite_frequency(integer)
                or half == integer
            ):
                raise RecipeNeedsParameters(
                    (
                        MissingParameter(
                            "flux_range", "No distinct calibrated flux endpoints"
                        ),
                    )
                )
            for key in ("start", "stop"):
                state = _cfg_object(inputs[key])
                raw = state["raw"]
                if (
                    state["mode"] != "expression"
                    or not isinstance(raw, str)
                    or set(re.findall(r"\bflx_(?:half|int)\b", raw))
                    != {"flx_half", "flx_int"}
                ):
                    raise RecipeNeedsParameters(
                        (
                            MissingParameter(
                                "flux_range", "No GUI calibration-derived range"
                            ),
                        )
                    )
        self.set_sweep(field, start=start, stop=stop, expts=expts)
        inputs = self._sweep_inputs(field)
        resolved_start = _cfg_object(inputs["start"])["resolved"]
        resolved_stop = _cfg_object(inputs["stop"])["resolved"]
        if (
            not _finite_frequency(resolved_start)
            or not _finite_frequency(resolved_stop)
            or resolved_start == resolved_stop
        ):
            raise GuiRpcError("Invalid resolved flux range", reason="invalid_cfg")
        self._binding.origins[field] = (
            "gui_calibration" if start is None else "explicit"
        )

    def _flux_unit(
        self, snapshot: dict[str, object], requested: str | None, *, explicit: bool
    ) -> str:
        unit = snapshot["unit"]
        info = _cfg_object(snapshot.get("info", {}))
        native = (
            unit == "none"
            and snapshot.get("type_name") == "FakeDevice"
            and info.get("type") == "FakeDevice"
            and requested == "native"
        )
        if native:
            return "native"
        if (
            snapshot.get("type_name") == "FakeDevice"
            or info.get("type") == "FakeDevice"
        ):
            raise GuiRpcError(
                "Fake flux requires confirmed native coordinates",
                reason="invalid_device",
            )
        if requested is not None and (requested != unit or requested == "native"):
            raise GuiRpcError(
                "Flux unit does not match the device coordinate",
                reason="invalid_device",
            )
        if not isinstance(unit, str) or not unit.strip() or unit in ("none", "native"):
            if explicit:
                raise GuiRpcError(
                    "Flux device has no physical unit", reason="invalid_device"
                )
            raise RecipeNeedsParameters(
                (MissingParameter("flux_device", "Flux device has no physical unit"),)
            )
        return unit

    def use_flux_device(
        self, name: str | None = None, *, unit: str | None = None
    ) -> None:
        """Select explicit name or device.flux.name; assert unit without conversion.

        Missing default source raises RecipeNeedsParameters. Unknown devices and
        unit mismatches fail as invalid_device. native is allowed only for a
        confirmed FakeDevice with original unit none and explicit unit="native".
        """
        if name is not None and (not name or not name.strip()):
            raise ValueError("flux device name must be non-empty")
        if unit is not None and (not unit or not unit.strip()):
            raise ValueError("flux unit must be non-empty")
        self._node("dev.flux_dev")
        device: object = name
        if name is None:
            values = self._binding.tools.send_gui_rpc("value.list", {})["values"]
            if any(_cfg_object(value)["key"] == "device.flux.name" for value in values):
                device = self._binding.tools.send_gui_rpc(
                    "value.read", {"key": "device.flux.name"}
                )["value"]
        if not isinstance(device, str) or not device.strip():
            raise RecipeNeedsParameters(
                (MissingParameter("flux_device", "No registered flux device source"),)
            )
        observed = self._binding.tools.send_gui_rpc("device.snapshot", {"name": device})
        resolved_unit = self._flux_unit(
            _cfg_object(observed["snapshot"]), unit, explicit=name is not None
        )
        self.set("dev.flux_dev", device)
        self._binding.origins["dev.flux_dev"] = (
            "device.flux.name" if name is None else "explicit"
        )
        self._binding.flux_unit = resolved_unit

    def set_sweep(
        self,
        field: str,
        *,
        start: float | None = None,
        stop: float | None = None,
        expts: int | None = None,
    ) -> None:
        """Edit only supplied sweep fields in their native unit; retain omitted ones.

        Supplied endpoints must be finite and not boolean. expts is an integer
        of at least 2; None retains the GUI count. Invalid explicit inputs and
        unknown fields raise ValueError. GUI field errors raise GuiRpcError.
        """
        _check_points(expts)
        for endpoint in (start, stop):
            if endpoint is not None and not _finite_frequency(endpoint):
                raise ValueError("sweep endpoints must be finite and not boolean")
        values: dict[str, object] = {}
        for key, value in (("start", start), ("stop", stop), ("expts", expts)):
            if value is not None:
                values[key] = value
        self._write_sweep(field, values)
        self._binding.origins[field] = "explicit" if values else "gui_default"
        for key in ("start", "stop", "expts"):
            self._binding.origins[f"{field}.{key}"] = (
                "explicit" if key in values else "gui_default"
            )

    def run(self) -> RunOperation:
        """Immediately send Run with captured cfg ref and return its yield handle.

        Capture actual conditions and result provenance. A pending between-yield
        cancel suppresses this one start and is consumed once. GUI rejection fails
        immediately; completion failures are thrown by the driver at the yield.
        """
        raise NotImplementedError("recipe tab implementation is not prepared")

    def accept(self, items: Sequence[str] | None = None) -> WritebackReceipt:
        """Write current draft items by stable target_name, Primary then Post.

        None selects all, an empty sequence selects none. Reject unknown, duplicate
        or cross-stage ambiguous names before any write. Ignore GUI checkboxes and
        resolve current IDs, not proposal snapshot IDs. Return confirmed progress;
        raise WritebackError with its receipt on the first failed stage. No refresh
        of guards, rollback, retry or matching-to-question-version is performed.
        """
        # The owner imports the receipt contracts here, so resolve it at execution.
        from zcu_tools.mcp.measure.writeback import write_current_draft

        receipt = write_current_draft(self._binding.tools, self._binding.tab, items)
        if receipt["status"] == "failed":
            raise WritebackError(receipt)
        return receipt


class RecipeRun:
    """Opaque capture of one tab Run, including a cancelled Run's partial result.

    The framework creates this handle at a RunOperation yield. The recipe decides
    whether to save partial data; cancellation does not make raw data saveable.
    """

    def save_raw(self) -> RawSaveReceipt:
        """Save this Run's raw data synchronously and return its actual receipt.

        Reject superseded or unavailable source data; await an admitted save's
        true outcome even when cancel arrives. Failures retain confirmed prefix
        facts and propagate. A reserved path is never reported as a saved path.
        """
        raise NotImplementedError("recipe run implementation is not prepared")

    def analyze(
        self,
        stage: AnalysisStage,
        *,
        params: Mapping[str, RecipeParameter] | None = None,
    ) -> AnalyzeOperation:
        """Immediately start primary/post analysis bound to this Run's source.

        params supplies named GUI analysis parameter updates; None keeps defaults.
        Post uses this Run's completed Primary source. Pending between-yield cancel
        suppresses this one start and is consumed once. Rejection fails immediately;
        the yielded handle delivers cancellation or throws completion failures.
        """
        raise NotImplementedError("recipe run implementation is not prepared")

    def propose_writeback(
        self, items: Sequence[str] | None = None
    ) -> WritebackQuestion:
        """Capture selected candidates from completed analyses without another read.

        items uses stable target_name, None selects all, empty selects none. Reject
        unknown, repeated or cross-stage ambiguous names before publishing the
        question. Yield the handle to await accepted/skipped; no timeout or write
        is implied. The recipe must call tab.accept explicitly to write anything.
        """
        raise NotImplementedError("recipe run implementation is not prepared")


RecipeOperation = RunOperation | AnalyzeOperation | WritebackQuestion
RecipeDelivery = tuple[
    RecipeRun | RecipeAnalysis | WritebackDecision | None, StepStatus
]
RecipeGenerator = Generator[RecipeOperation, RecipeDelivery, None]


@dataclass(frozen=True)
class RecipeDefinition:
    """One explicitly injected recipe and its operator-facing declarations.

    name is its unique non-empty MCP tool name; description explains its workflow.
    input_schema is its handwritten schema and must match run's typed keyword
    parameters. run receives RecipeSession first and returns RecipeGenerator.
    adapter_name is the public GUI adapter ID for guide lookup. summary_parameters
    selects captured actual conditions; summary_estimates selects native scalar
    estimates. Empty summary tuples are valid. The registry validates definitions
    and schemas before any execution; this value does not register on import.
    """

    name: str
    description: str
    input_schema: RecipeInputSchema
    run: Callable[..., RecipeGenerator]
    adapter_name: str = field(kw_only=True)
    summary_parameters: tuple[SummaryParameter, ...] = field(kw_only=True)
    summary_estimates: tuple[SummaryEstimate, ...] = field(kw_only=True)
