"""Tab editing and Run preparation for author-facing generator recipes."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from typing import Literal, TypedDict, TypeGuard

from pydantic import TypeAdapter

from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.remote.form_description import (
    RecipeElementsWire,
    RecipeFieldObservation,
)
from zcu_tools.mcp.measure.execution_reply import SummaryElements
from zcu_tools.mcp.measure.recipe_capture import (
    ActualField,
    SweepSources,
    capture_actual,
)
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeDeviceState,
    RecipeExpression,
    RecipeFieldEdit,
    RecipeParameter,
    RecipeRunSnapshot,
    WritebackError,
    WritebackReceipt,
    recipe_gui_object,
)
from zcu_tools.mcp.measure.recipe_form import (
    RecipeFieldPath,
    RecipeFormField,
    RecipeModuleValues,
    directed_errors,
    field_observation,
    field_path,
    read_form_field,
    read_module_values,
)
from zcu_tools.mcp.measure.recipe_handles import (
    RecipeRunBinding,
    RecipeStep,
    RecipeStepCancelled,
    RecipeTabBinding,
    RunOperation,
    RunStepResult,
)
from zcu_tools.mcp.measure.session import GuiRpcError
from zcu_tools.mcp.measure.writeback import write_current_draft


class _CfgEdit(TypedDict):
    """One wire edit: path is a public field path; value is GUI input data."""

    path: list[str | int]
    value: object


def _wire_value(value: object) -> YamlValue:
    """Copy the JSON/YAML representation, retaining opaque tagged objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _wire_value(item) for key, item in recipe_gui_object(value).items()
        }
    raise GuiRpcError("Invalid GUI value", reason="incompatible_wire")


def _is_expression(value: object) -> bool:
    """Recognize only the existing one-key expression wire carrier."""
    return (
        isinstance(value, dict)
        and set(value) == {"expr"}
        and isinstance(value["expr"], str)
    )


def _check_edit_value(value: object) -> None:
    if (
        value is None
        or isinstance(value, (str, bool, int, float))
        or _is_expression(value)
    ):
        return
    if isinstance(value, list) and all(
        item is None or isinstance(item, (str, bool, int, float)) for item in value
    ):
        return
    raise ValueError("value must be a scalar, scalar list or RecipeExpression")


def _source_label(source: object) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty provenance label")
    return source


def _check_points(expts: object) -> None:
    if expts is not None and (
        not isinstance(expts, int) or isinstance(expts, bool) or expts < 2
    ):
        raise ValueError("expts must be an integer of at least 2")


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


class RecipeTab:
    """Opaque tab handle returned by RecipeSession.open_tab.

    Fields are nonempty tuples of str names and int original list indexes.
    Numeric strings remain literal field names; names never select positions.
    Methods read current field observations and record parameter sources.
    GUI errors propagate without re-read, retry or rollback. Authors obtain
    this handle from open_tab rather than constructing the private binding.
    """

    def __init__(self, binding: RecipeTabBinding) -> None:
        self._binding = binding

    def read_value(self, path: str) -> YamlValue:
        """Read one non-blank logical leaf from this tab's fixed Reset capture.

        path is a parameter path such as $custom.threshold or Q1.ext.seed, not an
        arithmetic formula. Missing and explicit null both return None. Each call
        uses tab.parameter_read without overrides and returns a detached value.
        It does not change cfg, provenance or Run state, or refresh guards. Source,
        alias and fixed-connection failures propagate without fallback or retry.
        ValueError rejects blank paths; malformed wire raises incompatible_wire.
        """
        if not path or not path.strip():
            raise ValueError("path must be a non-empty logical leaf path")
        return _wire_value(self._read_parameter(path))

    def read_library_root(self, reference: str | None) -> str | None:
        """Ask the source owner for reference's canonical library module root.

        None returns None without an RPC. Other references must be non-blank;
        custom/missing modules return None, never a client-guessed root. Queries
        use the fixed Reset capture, not context.snapshot or live aliases. Cfg,
        provenance and Run state are unchanged. Source/connection errors propagate
        without fallback/retry; malformed root replies raise incompatible_wire.
        """
        if reference is not None and (not reference or not reference.strip()):
            raise ValueError("reference must be a non-empty module reference")
        return self._library_reference(reference)

    def read_field(self, field: RecipeFieldPath) -> RecipeFormField:
        """Read a logical field with one fresh tab.read_recipe_field RPC.

        field is a nonempty typed logical path; only generated wrappers are
        hidden. Range count is n. Metadata, authored input, ordinary value,
        value_status, selector/source identity, readonly and local errors come
        from one owner turn. Ref reads project fixed captured Specs without
        materialization, source refresh or write authority. Unknown/inactive
        paths and source failures retain owner errors; malformed local paths
        raise ValueError, malformed replies raise incompatible_wire. Returned
        views are detached. Cfg validity still belongs to get_cfg/Run admission.
        """
        return self._node(field)

    def read_device(self, name: str) -> RecipeDeviceState:
        """Read name's cached device.snapshot once on the fixed GUI connection.

        name is a non-blank registered device name. This does not connect, setup,
        poll hardware, change cfg/provenance, or interpret/convert units. Missing
        unit or malformed wire raises GuiRpcError(incompatible_wire). Owner errors
        (including unknown devices) and connection failures propagate, without
        reconnect or retry. ValueError rejects blank names before dispatch.
        """
        if not name or not name.strip():
            raise ValueError("device name must be non-empty")
        reply = self._binding.tools.send_gui_rpc("device.snapshot", {"name": name})
        snapshot = recipe_gui_object(reply.get("snapshot"))
        if "unit" not in snapshot:
            raise GuiRpcError("Missing device unit", reason="incompatible_wire")
        return RecipeDeviceState(unit=_wire_value(snapshot["unit"]))

    def _read_parameter(self, path: str) -> object:
        """Read one logical path, retaining missing/null values."""
        reply = self._binding.tools.send_gui_rpc(
            "tab.parameter_read",
            {
                "tab_id": self._binding.tab,
                "paths": [path],
                "module_refs": [],
            },
        )
        return recipe_gui_object(reply.get("values")).get(path)

    def _library_reference(self, reference: str | None) -> str | None:
        """Resolve a canonical module root using the fixed Reset sources."""
        if reference is None:
            return None
        reply = self._binding.tools.send_gui_rpc(
            "tab.parameter_read",
            {
                "tab_id": self._binding.tab,
                "paths": [],
                "module_refs": [reference],
            },
        )
        modules = recipe_gui_object(reply.get("modules"))
        if reference not in modules:
            raise GuiRpcError("Missing module lookup reply", reason="incompatible_wire")
        root = modules[reference]
        if root is not None and not isinstance(root, str):
            raise GuiRpcError("Invalid module lookup reply", reason="incompatible_wire")
        return root

    def _node(self, name: RecipeFieldPath) -> RecipeFormField:
        path = tuple(field_path(name))
        return read_form_field(self._describe(path), path)

    def _describe(self, path: RecipeFieldPath) -> RecipeFieldObservation:
        return field_observation(
            self._binding.tools.send_gui_rpc(
                "tab.read_recipe_field",
                {"tab_id": self._binding.tab, "path": list(path)},
            )
        )

    def _edit(self, edits: list[_CfgEdit]) -> Mapping[str, object] | None:
        if edits:
            return self._binding.tools.send_gui_rpc(
                "tab.edit_recipe_fields",
                {
                    "tab_id": self._binding.tab,
                    "edits": edits,
                },
            )

        return None

    def use_library(
        self, field: RecipeFieldPath, name: str | None, *, required: str | None = None
    ) -> MissingParameter | None:
        """Select an actual library key using one logical owner command.

        field is a logical generated selector. name is a nonblank actual key;
        owner checks captured family/choices/locks/ref ancestors. None retains
        input and performs one fresh read. required names a nonblank caller
        parameter; absent/invalid retained keys return MissingParameter without
        mutation or success provenance. Explicit success records explicit,
        retained success gui_default. No Option/input inspection, materialization,
        refresh or retry occurs; owner and connection failures propagate.
        """
        field_path(field)
        if required is not None and not required.strip():
            raise ValueError("required must be a nonempty parameter name")
        if name is not None:
            if not name.strip():
                raise ValueError("library name must be non-empty")
            selected = self._select_module(field, "use_library", key=name)
        else:
            selected = self.read_field(field)
        if selected.selector is None:
            raise ValueError(f"Not a module reference field: {field}")
        valid = (
            selected.mode == "key"
            and selected.key is not None
            and selected.value_status == "available"
            and not selected.errors
        )
        if name is not None and (not valid or selected.key != name):
            raise GuiRpcError(f"Invalid {field} reference", reason="invalid_cfg")
        if name is None and required is not None and not valid:
            return MissingParameter(required, f"Provide a valid {field} library key")
        self._binding.origins[field_path(field)] = (
            "explicit" if name is not None else "gui_default"
        )
        return None

    def _select_module(
        self,
        field: RecipeFieldPath,
        action: Literal["use_library", "use_inline", "disable"],
        *,
        key: str | None = None,
    ) -> RecipeFormField:
        path = tuple(field_path(field))
        params: dict[str, object] = {
            "tab_id": self._binding.tab,
            "path": list(path),
            "action": action,
        }
        if key is not None:
            params["key"] = key
        reply = self._binding.tools.send_gui_rpc("tab.select_recipe_module", params)
        return read_form_field(
            field_observation(recipe_gui_object(reply.get("field"))), path
        )

    def disable_library(self, field: RecipeFieldPath) -> None:
        """Disable a nullable logical selector; owner checks locks and ancestry.

        Sends one selector command, never set(null). On success records disabled
        provenance. Admission/source/control failures propagate without retry.
        """
        selected = self._select_module(field, "disable")
        if (
            selected.selector is None
            or selected.selector["mode"] != "disabled"
            or selected.errors
        ):
            raise GuiRpcError(f"Cannot disable {field}", reason="invalid_cfg")
        self._binding.origins[field_path(field)] = "disabled"

    def read_module(self, field: RecipeFieldPath) -> RecipeModuleValues:
        """Read one core module's value, mode and source from fixed Reset sources.

        field is a nonempty typed tuple of str core module field names and int
        original model-list indexes, not an inline leaf. Numeric strings remain
        literal names. Paths are forwarded unchanged to the shared owner. Values
        are detached module JSON, without full-cfg validation, native to_cfg or
        extras lowering. Other invalid fields do not block this read. The owner
        rejects this module's invalid values with NOT_VALID, unknown/nonmodule
        paths with invalid_params, and revoked resources with RESOURCE_GONE.
        Busy and notification reads are allowed. No input change, publication,
        source refresh or retry occurs. GUI/source/connection failures propagate.
        """
        return read_module_values(
            self._binding.tools.send_gui_rpc(
                "tab.read_module",
                {"tab_id": self._binding.tab, "field": list(field_path(field))},
            )
        )

    def use_inline(self, field: RecipeFieldPath) -> None:
        """Explicitly copy a logical selector from its fixed captured Spec.

        Sends tab.select_recipe_module/use_inline even for existing inline, so
        holder/busy/lifetime/notification guards still apply to no-ops. Outer ref
        ancestors must be copied explicitly first. Missing/disabled/unresolved
        input never creates defaults. A successful copy preserves child keys;
        later failed writes do not roll it back. Owner/source/connection failures
        propagate without refresh, fallback or retry.
        """
        self._select_module(field, "use_inline")

    def retain_module_source(
        self,
        field: RecipeFieldPath,
        *,
        summary_field: RecipeFieldPath,
        value_path: RecipeFieldPath,
    ) -> None:
        """Retain provenance for one logical scalar without materializing a module.

        field is a logical selector; value_path is its nonempty relative logical
        scalar path. summary_field must identify that same leaf. Validates its
        current ordinary value/status/errors, then records library:<actual-key>
        from owner source or field for authored inline values. No input changes
        or write authority result. Fresh Run reads capture detached facts.
        """
        path = tuple(field_path(field))
        value_path = field_path(value_path)
        if tuple(field_path(summary_field)) != (*path, *value_path):
            raise ValueError("summary_field must match the logical module value path")
        node = self.read_field(summary_field)
        if (
            node.selector is not None
            or node.fields
            or node.value_status != "available"
            or node.errors
        ):
            raise ValueError("Module value path must select an available scalar")
        self._binding.origins[field_path(summary_field)] = (
            f"library:{node.source['key']}" if node.source is not None else repr(field)
        )

    def retain_source(self, field: RecipeFieldPath, source: str) -> None:
        """Record a caller-chosen provenance label for an unchanged cfg field.

        field is an existing typed scalar or sweep path. source is a nonblank
        string, such as a library label; this method interprets no domain policy.
        It changes only this tab's source metadata for the next Run. Subsequent
        set/set_sweep calls can replace the label, and captured Runs remain fixed.
        One fresh logical read validates the field category. No Form input
        change or validation of the resolved value occurs.
        Unknown/non-scalar/non-sweep fields and empty labels raise ValueError before metadata changes. set(None) still records gui_default.
        """
        label = _source_label(source)
        node = self._node(field)
        if node.selector is not None or (
            node.fields and not {"start", "stop", "n"} <= set(node.fields)
        ):
            raise ValueError(f"Not a scalar or sweep cfg field: {field}")
        self._binding.origins[field_path(field)] = label

    def set_fields(self, edits: Sequence[RecipeFieldEdit]) -> None:
        """Send supplied logical scalar edits in one tab.edit_recipe_fields RPC.

        edits is an ordered sequence of typed scalar paths, direct/expression
        inputs and non-blank provenance labels. None values retain current inputs
        and record gui_default. Empty/default-only batches send no edit. Validate
        local path syntax, carrier and label before any mutation; local
        errors raise ValueError. Expressions go unchanged to the GUI owner.
        The resource returns changed and complete logical Form errors. Check directed
        errors before recording sources; invalid replies raise invalid_cfg
        but leave the owner's committed draft intact.
        GUI/connection failures propagate without refresh, retry or rollback;
        the resource owns the atomic write.
        """
        planned = tuple(edits)
        wire: list[_CfgEdit] = []
        for edit in planned:
            if not edit.source or not edit.source.strip():
                raise ValueError("source must be a non-empty provenance label")
            field_path(edit.field)
            _check_edit_value(edit.value)
            if edit.value is not None:
                wire.append(
                    {
                        "path": list(field_path(edit.field)),
                        "value": deepcopy(edit.value),
                    }
                )
        reply = self._edit(wire)
        for edit in planned:
            if (
                reply is not None
                and edit.value is not None
                and directed_errors(reply, tuple(field_path(edit.field)))
            ):
                raise GuiRpcError(
                    f"Invalid {edit.field} value; owner retained the draft",
                    reason="invalid_cfg",
                )
        for edit in planned:
            self._binding.origins[field_path(edit.field)] = (
                edit.source if edit.value is not None else "gui_default"
            )

    def set(
        self,
        field: RecipeFieldPath,
        value: RecipeParameter | RecipeExpression,
        *,
        source: str = "explicit",
    ) -> None:
        """Set one typed scalar path to a direct input or GUI expression carrier.

        None keeps its current input and records gui_default. source is a non-blank
        provenance label for supplied values. Unknown/non-scalar fields and invalid
        carriers/labels are checked locally. The GUI owns path/admission errors,
        parses expressions
        and validates values; returned errors raise GuiRpcError(invalid_cfg) without
        retry or rollback. Earlier returned field views remain detached.
        """
        self.set_fields((RecipeFieldEdit(field, value, source),))

    def set_integer_points(
        self, field: RecipeFieldPath, values: list[int] | None
    ) -> None:
        """Replace one Log/List integer field with a complete explicit List input.

        field is a nonempty typed logical path to a LogPoints-authored field.
        values is a list of built-in integers, preserving order and duplicates;
        None retains even an invalid GUI draft and records gui_default. Supplied
        values atomically replace the whole branch, without sampling or conversion.
        Experiment bounds, nonempty-list and readonly rules belong to the owner.

        Invalid path syntax or noninteger values raise ValueError before editing.
        GUI structural/admission errors propagate as GuiRpcError. Owner validation
        errors raise GuiRpcError(invalid_cfg), retaining the invalid List draft
        without recording successful provenance. No new RPC or source refresh occurs.
        """
        path = field_path(field)
        if values is None:
            self._binding.origins[path] = "gui_default"
            return
        if type(values) is not list or any(type(value) is not int for value in values):
            raise ValueError("values must be a list of integers or null")
        reply = self._edit(
            [{"path": list(path), "value": {"kind": "list", "values": [*values]}}]
        )
        if reply is not None and directed_errors(reply, path):
            raise GuiRpcError(
                f"Invalid {field} integer points; draft retained", reason="invalid_cfg"
            )
        self._binding.origins[path] = "explicit"

    def _write_sweep(self, field: RecipeFieldPath, values: dict[str, object]) -> None:
        field_path(field)
        if not values:
            return
        edits: list[_CfgEdit]
        if {"start", "stop", "expts"} <= values.keys():
            edits = [
                {
                    "path": list(field_path(field)),
                    "value": {
                        "kind": "linrange",
                        "start": values["start"],
                        "stop": values["stop"],
                        "n": values["expts"],
                    },
                }
            ]
        else:
            edits = [
                {
                    "path": [*field_path(field), "n" if key == "expts" else key],
                    "value": value,
                }
                for key, value in values.items()
            ]
        reply = self._edit(edits)
        if reply is not None and directed_errors(reply, tuple(field_path(field))):
            raise GuiRpcError(
                f"Invalid {field} sweep; draft retained", reason="invalid_cfg"
            )

    def set_sweep(
        self,
        field: RecipeFieldPath,
        *,
        start: float | RecipeExpression | None = None,
        stop: float | RecipeExpression | None = None,
        expts: int | None = None,
        sources: SweepSources | None = None,
    ) -> None:
        """Set sweep coordinates in native units; complete values select Range + Expts.

        Supplying start, stop and expts replaces the whole sweep atomically,
        independent of active coordinate mode or an invalid previous draft.
        Partial values edit only active Range leaves, retaining omitted ones;
        inactive paths reject rather than guessing a conversion.

        Supplied numeric endpoints must be finite and not boolean; RecipeExpression
        endpoints pass unchanged to the GUI for parsing/resolution/validation.
        expts is an integer
        of at least 2; None retains the GUI count. sources optionally labels
        start/stop/expts separately in the Run capture; all three non-empty labels
        are required. Omitted values always retain gui_default provenance.
        Without sources, capture uses the existing explicit/gui_default labels.
        Invalid explicit inputs, sources and unknown fields raise ValueError.
        GUI field errors raise GuiRpcError.
        """
        if sources is not None and (
            set(sources) != {"start", "stop", "expts"}
            or any(
                not isinstance(label, str) or not label.strip()
                for label in sources.values()
            )
        ):
            raise ValueError(
                "sources must label start, stop and expts with non-empty strings"
            )
        _check_points(expts)
        for endpoint in (start, stop):
            if (
                endpoint is not None
                and not _is_expression(endpoint)
                and not _finite_number(endpoint)
            ):
                raise ValueError("sweep endpoints must be finite and not boolean")
        values: dict[str, object] = {}
        for key, value in (("start", start), ("stop", stop), ("expts", expts)):
            if value is not None:
                values[key] = value
        self._write_sweep(field, values)
        captured_sources: SweepSources = (
            deepcopy(sources)
            if sources is not None
            else {"start": "explicit", "stop": "explicit", "expts": "explicit"}
        )
        for key in ("start", "stop", "expts"):
            if key not in values:
                captured_sources[key] = "gui_default"
            model_field = "n" if key == "expts" else key
            self._binding.origins[(*field_path(field), model_field)] = captured_sources[
                key
            ]
        self._binding.origins[field_path(field)] = captured_sources

    def run_step(self) -> Generator[RecipeStep, None, RunStepResult]:
        """Send Run and await its named result using ``yield from tab.run_step()``.

        Uses the same cfg, source and cancellation contract as run(). Completion
        failures are thrown here. Cancelled results let the author clean up and
        return; finish_early retains the partial Run for saving and analysis.
        """
        operation = self.run()
        yield RecipeStep(operation)
        if operation.delivery is None:
            raise ValueError("Run helper resumed without driver delivery")
        return operation.delivery

    def run(self) -> RunOperation:
        """Immediately send Run and return an opaque handle for framework use.

        Recipe authors use yield from run_step(), not yield run().

        The recipe cfg publication must be valid. Reset captured aliases and
        parameter sources before edits; Run sends no source overrides.
        Capture actual conditions and result provenance. A pending between-yield
        cancel suppresses this one start and is consumed once. GUI rejection fails
        immediately; completion failures are thrown by the driver at the yield.
        """
        suppressed = RunOperation(None)
        self._binding.observe_run(suppressed)
        if self._binding.closed.is_set():
            raise GuiRpcError("MCP session is closed", reason="session_closed")
        consume_cancel = self._binding.consume_cancel
        if consume_cancel is not None and consume_cancel():
            return suppressed
        current = self._binding.tools.send_gui_rpc(
            "tab.get_cfg", {"tab_id": self._binding.tab}
        )
        if current["errors"]:
            raise GuiRpcError("Recipe cfg is not Valid", reason="invalid_cfg")
        actual = capture_actual(self._actual_fields())
        binding = RecipeRunBinding(
            self._binding, RecipeRunSnapshot(self._binding.tab, actual)
        )
        operation = RunOperation(binding)
        self._binding.observe_run(operation)
        tools = self._binding.tools
        tab = self._binding.tab
        tools.send_gui_rpc("tab.snapshot", {"tab_id": tab})
        tools.send_gui_rpc("soc.info", {"include_cfg": True})
        for device in tools.send_gui_rpc("device.list", {})["devices"]:
            tools.send_gui_rpc("device.snapshot", {"name": device["name"]})

        def admit() -> None:
            if self._binding.closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            consume_cancel = self._binding.consume_cancel
            if consume_cancel is not None and consume_cancel():
                raise RecipeStepCancelled
            binding.publish(replace(binding.capture, start_status="unknown"))

        payload: dict[str, object] = {"tab_id": tab}
        try:
            started = tools.gui.send_gui_rpc(
                "tab.run_start", payload, before_send=admit
            )
        except RecipeStepCancelled:
            self._binding.observe_run(suppressed)
            return suppressed
        except GuiRpcError as error:
            if error.request_rejected:
                binding.publish(
                    replace(
                        binding.capture,
                        start_status="not_started",
                        start_reason=error.reason or error.code,
                    )
                )
            raise
        binding.publish(
            replace(binding.capture, op=started["handle"], start_status="running")
        )
        return operation

    def _capture_cfg_field(
        self, name: RecipeFieldPath, source: str | SweepSources
    ) -> ActualField:
        return self._field_fact(name, self._node(name), source)

    @staticmethod
    def _field_fact(
        name: RecipeFieldPath, node: RecipeFormField, source: str | SweepSources
    ) -> ActualField:
        if node.value_status != "available" or node.errors:
            raise GuiRpcError(f"Unavailable {name} summary value", reason="invalid_cfg")
        value = node.value
        if node.mode == "key":
            value = node.key
        elif isinstance(value, dict) and {"start", "stop", "n"} <= value.keys():
            value = {
                "start": value["start"],
                "stop": value["stop"],
                "expts": value["n"],
            }
        fact = ActualField(value=value, source=source)
        if node.element is not None:
            fact["element"] = node.element
        return fact

    def _actual_fields(self) -> dict[str, ActualField]:
        fields: dict[str, ActualField] = {}
        for parameter in self._binding.summary_parameters:
            if isinstance(parameter, SummaryElements):
                fields.update(self._element_facts(parameter))
                continue
            if parameter.field not in self._binding.origins:
                continue
            source = self._binding.origins[parameter.field]
            fact = self._capture_cfg_field(parameter.field, source)
            if parameter.unit is not None:
                fact["unit"] = parameter.unit
            fields[parameter.name] = fact
        return fields

    def _element_facts(self, declaration: SummaryElements) -> dict[str, ActualField]:
        reply = self._binding.tools.send_gui_rpc(
            "tab.read_elements",
            {
                "tab_id": self._binding.tab,
                "path": list(declaration.field),
                "fields": [
                    list(parameter.field) for parameter in declaration.parameters
                ],
            },
        )
        try:
            observed = TypeAdapter(RecipeElementsWire).validate_python(
                reply, strict=True
            )
        except ValueError as error:
            raise GuiRpcError(
                "Invalid element observations", reason="incompatible_wire"
            ) from error
        facts: dict[str, ActualField] = {}
        for row in observed["elements"]:
            for parameter, raw in zip(
                declaration.parameters, row["fields"], strict=True
            ):
                path = (*declaration.field, row["index"], *parameter.field)
                node = read_form_field(raw, path)
                source = self._binding.origins.get(path, declaration.source)
                fact = self._field_fact(path, node, source)
                if parameter.unit is not None:
                    fact["unit"] = parameter.unit
                facts[f"{declaration.name}/{row['index']}/{parameter.name}"] = fact
        return facts

    def accept(self, items: Sequence[str] | None = None) -> WritebackReceipt:
        """Write current draft items by stable target_name, Primary then Post.

        None selects all, an empty sequence selects none. Reject unknown, duplicate
        or cross-stage ambiguous names before any write. Ignore GUI checkboxes and
        resolve current IDs, not proposal snapshot IDs. Return confirmed progress;
        raise WritebackError with its receipt on the first failed stage. No refresh
        of guards, rollback, retry or matching-to-question-version is performed.
        """
        receipt = write_current_draft(self._binding.tools, self._binding.tab, items)
        self._binding.observe_writeback(receipt)
        if receipt["status"] == "failed":
            raise WritebackError(receipt)
        return receipt
