"""Recipe Form primitives through registered synthetic recipes and GUI wire."""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    GaussWaveformCfg,
    PulseSpec,
)
from zcu_tools.gui.app.measure.remote.form_description import recipe_field_wire
from zcu_tools.gui.app.measure.remote.form_errors import recipe_form_errors
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.form_adapter import Form, LogicalEdit
from zcu_tools.mcp.measure import recipe_api as api
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from ._script_recipe_support import Inputs, Report, SyntheticGui, start
from ._support import make_client


class ReadTone(ConfigBase):
    gain: float = 0.2


class PrimitiveCfg(ConfigBase):
    pulse: PulseCfg | None = None
    required: PulseCfg
    count: int = Field(default=1, gt=0)
    mode: Literal["fast", "slow"] = "fast"
    sweep: SweepCfg
    payload: list[list[dict[str, int]]] = Field(default_factory=list)
    label: str | None = "seed"
    tones: list[ReadTone] = Field(default_factory=lambda: [ReadTone(), ReadTone()])


class WireEdit(BaseModel):
    path: tuple[str | int, ...]
    value: JsonValue


type Responder = Callable[[str, dict[str, object]], dict[str, object]]


@pytest.fixture
def form_owner() -> tuple[TabFormResource, Responder]:
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={
                            "drive": PulseSpec(
                                ch=1,
                                nqz=1,
                                freq=4100,
                                gain=0.3,
                                waveform="Q.waveforms.gauss",
                            )
                        },
                        waveforms={
                            "gauss": GaussWaveformCfg(length=0.8, sigma=0.2),
                            "const": ConstWaveformCfg(length=0.1),
                        },
                    )
                },
            )
        )
    )
    owner = TabFormResource(
        lambda _ctx: Form(PrimitiveCfg, {}),
        capture=lambda _aliases: TabFormSources(env),
        mutation_allowed=lambda: True,
    )
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        with recipe_form_errors():
            if method == "tab.edit_recipe_fields":
                edits = TypeAdapter(list[WireEdit]).validate_python(params["edits"])
                changed = owner.edit_recipe_fields(
                    [LogicalEdit(e.path, e.value) for e in edits]
                )
                return {"changed": changed, "errors": []}
            if method == "tab.select_recipe_option":
                option = TypeAdapter(str | None).validate_python(params["option"])
                return {
                    "changed": owner.select_recipe_option(
                        tuple(
                            TypeAdapter(list[str | int]).validate_python(params["path"])
                        ),
                        option,
                    )
                }
            if method == "tab.run_recipe_command":
                return {
                    "changed": owner.run_recipe_command(
                        tuple(
                            TypeAdapter(list[str | int]).validate_python(params["path"])
                        ),
                        str(params["name"]),
                    )
                }
            if method == "tab.check_recipe_ready":
                errors = owner.recipe_ready_errors()
                return {
                    "errors": [
                        {"path": list(p), "message": m}
                        for p, m in (errors.messages.items() if errors else ())
                    ]
                }
            if method == "tab.read_recipe_field":
                return dict(
                    recipe_field_wire(
                        owner.read_recipe_field(
                            tuple(
                                TypeAdapter(list[str | int]).validate_python(
                                    params["path"]
                                )
                            )
                        )
                    )
                )
        return gui.respond(method, params)

    return owner, respond


def test_select_and_command_follow_owner_options_and_captured_sources(
    tmp_path: Path,
    form_owner: tuple[TabFormResource, Responder],
) -> None:
    owner, respond = form_owner

    @api.recipe("synthetic", "Use Form choices and explicit copy commands")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        tab.select("pulse", "Q.pulses.drive")
        tab.command("pulse", "to_inline")
        assert tab.read("pulse.gain").value == 0.3
        assert tab.read("pulse.waveform").source == api.FormSource(
            "Q.waveforms.gauss", "waveform", ()
        )
        tab.command("pulse.waveform", "to_inline")
        tab.command("pulse", "to_inline")
        tab.command("pulse.waveform", "to_inline")
        assert tab.read("pulse.gain").value == 0.3
        assert tab.read("pulse.waveform.sigma").value == 0.2
        tab.select("pulse.waveform", "Inline: const")
        assert tab.read("pulse.waveform.length").value == 0.8
        assert "sigma" not in tab.read("pulse.waveform").fields
        tab.select("mode", "slow")
        assert tab.read("mode").value == "slow"
        tab.select("pulse", None)
        assert tab.read("pulse").value is None
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        assert owner.read_recipe_field(("mode",)).value == "slow"
    finally:
        client.context.session.close()


def test_read_dotted_paths_observes_nested_fields_and_original_list_indexes(
    tmp_path: Path,
    form_owner: tuple[TabFormResource, Responder],
) -> None:
    _owner, respond = form_owner

    @api.recipe("synthetic", "Observe dotted logical field paths")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        tab.set("tones.1.gain", 0.7)
        tab.select("pulse", "Q.pulses.drive")
        assert tab.read("tones.0.gain").value == 0.2
        assert tab.read("tones.1.gain").value == 0.7
        assert tab.read("pulse.waveform.sigma").value == 0.2
        assert tab.read("count").value == 1
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
    finally:
        client.context.session.close()


def test_check_ready_reports_every_owner_error_without_mutation_or_run(
    tmp_path: Path,
    form_owner: tuple[TabFormResource, Responder],
) -> None:
    owner, respond = form_owner

    @api.recipe("synthetic", "Validate the complete current draft")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        tab.set("count", 0)
        tab.check_ready()
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        reply = client.call("synthetic", {"count": 9}).data
        assert reply["status"] == "needs_parameters", reply
        errors = reply["parameter_errors"]
        assert {tuple(e["path"]) for e in errors} == {
            ("count",),
            ("required",),
            ("sweep", "start"),
            ("sweep", "stop"),
            ("sweep", "n"),
        }
        assert all(e["message"] for e in errors)
        assert owner.read_recipe_field(("count",)).input == 0
        assert [m for m, _p in client.transport.sent if m.startswith("tab.")] == [
            "tab.new",
            "tab.reset_cfg",
            "tab.edit_recipe_fields",
            "tab.check_recipe_ready",
        ]
    finally:
        client.context.session.close()


@pytest.mark.parametrize(("path", "option"), [("required", None), ("mode", "unknown")])
def test_select_rejects_nonnullable_null_and_unknown_options_without_a_prefix(
    tmp_path: Path,
    form_owner: tuple[TabFormResource, Responder],
    path: str,
    option: str | None,
) -> None:
    owner, respond = form_owner
    before = owner.input.to_json()

    @api.recipe("synthetic", "Reject unavailable owner selections")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        tab.select(path, option)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert (
            "nullable" if option is None else "option"
        ) in snapshot.error.message.lower()
        assert owner.input.to_json() == before
    finally:
        client.context.session.close()


def test_named_reads_use_fixed_parameters_and_cached_device_units(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.parameter_read":
            return {
                "values": {"$qubit.freq": 4100.0, "$qubit.null": None},
                "modules": {},
            }
        if method == "device.snapshot":
            return {"snapshot": {"unit": "A"}}
        return gui.respond(method, params)

    @api.recipe("synthetic", "Read fixed parameters and a cached unit")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        assert tab.read_parameter("$qubit.freq") == 4100.0
        assert tab.read_parameter("$qubit.null") is None
        assert tab.read_parameter("$qubit.missing", missing_ok=True) is None
        assert session.device_unit("flux") == "A"
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        assert [m for m, _p in client.transport.sent if m.startswith("tab.")] == [
            "tab.new",
            "tab.reset_cfg",
            "tab.parameter_read",
            "tab.parameter_read",
            "tab.parameter_read",
        ]
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    "path", ["", "tones..gain", ".gain", "tones.-1.gain", "tones. gain"]
)
def test_override_paths_reject_malformed_dotted_input(path: str) -> None:
    with pytest.raises(ValidationError):
        api.FieldEdit(path=path, value=1)


def test_json_overrides_preserve_expression_sweep_nested_objects_and_null(
    tmp_path: Path,
    form_owner: tuple[TabFormResource, Responder],
) -> None:
    _owner, respond = form_owner

    @api.recipe("synthetic", "Pass structured JSON inputs intact to the Form owner")
    async def synthetic(session: api.RecipeSession, inputs: api.RecipeInputs) -> Report:
        tab = session.open_tab("synthetic")
        for edit in inputs.overrides:
            tab.set(edit.path, edit.value)
        assert tab.read("count").value == 5
        assert tab.read("sweep").value == {
            "kind": "linrange",
            "start": 10.0,
            "stop": 20.0,
            "n": 7,
        }
        assert tab.read("payload").value == [[{"count": 2}]]
        assert tab.read("label").value is None
        assert tab.read("label").value_status == "available"
        return Report(count=1)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context,
            "synthetic",
            {
                "overrides": [
                    {"path": "count", "value": {"expr": "2 + 3"}},
                    {
                        "path": "sweep",
                        "value": {
                            "kind": "linrange",
                            "start": 10.0,
                            "stop": 20.0,
                            "n": 7,
                        },
                    },
                    {"path": "payload", "value": [[{"count": 2}]]},
                    {"path": "label", "value": None},
                ],
            },
        )
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
    finally:
        client.context.session.close()


def test_structured_overrides_cannot_replace_generated_selectors(
    tmp_path: Path,
    form_owner: tuple[TabFormResource, Responder],
) -> None:
    owner, respond = form_owner
    before = owner.input.to_json()

    @api.recipe("synthetic", "Leave selector admission with the Form owner")
    async def synthetic(session: api.RecipeSession, inputs: api.RecipeInputs) -> Report:
        tab = session.open_tab("synthetic")
        for edit in inputs.overrides:
            tab.set(edit.path, edit.value)
        return Report(count=1)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context,
            "synthetic",
            {
                "overrides": [
                    {
                        "path": "pulse",
                        "value": {"kind": "ref", "name": "Q.pulses.drive"},
                    }
                ],
            },
        )
        execution.join()
        snapshot = execution.snapshot()
        assert snapshot.status == "failed"
        assert snapshot.error is not None
        assert "module selection" in snapshot.error.message
        assert owner.input.to_json() == before
    finally:
        client.context.session.close()


@pytest.mark.parametrize("reuse", [None, "tab-1"])
def test_open_tab_named_controls_reset_with_local_target(
    tmp_path: Path, reuse: str | None
) -> None:
    @api.recipe("synthetic", "Bind a local target and optionally reuse a tab")
    async def synthetic(session: api.RecipeSession, inputs: api.RecipeInputs) -> Report:
        session.open_tab(
            "synthetic",
            target={"qubit": inputs.target, "kept": api.KEEP, "global": None},
            reuse=inputs.reuse_tab_id,
        )
        return Report(count=1)

    client = make_client(tmp_path, SyntheticGui().respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {"target": "Q2", "reuse_tab_id": reuse}
        )
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        assert [p for m, p in client.transport.sent if m == "tab.reset_cfg"] == [
            {"tab_id": "tab-1", "aliases": {"qubit": "Q2"}},
        ]
        assert sum(m == "tab.new" for m, _p in client.transport.sent) == (
            1 if reuse is None else 0
        )
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    ("owner_error", "missing_ok", "expected"),
    [
        (None, False, "parameter_missing"),
        ("source_failure", True, "source_failure"),
        ("malformed_path", True, "malformed_path"),
    ],
)
def test_read_parameter_only_allows_owner_missing_leaves(
    tmp_path: Path,
    owner_error: str | None,
    missing_ok: bool,
    expected: str,
) -> None:
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
    from zcu_tools.mcp.measure.session import GuiRpcError

    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.parameter_read":
            if owner_error is not None:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS, "owner rejection", reason=owner_error
                )
            return {"values": {}, "modules": {}}
        return gui.respond(method, params)

    @api.recipe("synthetic", "Propagate parameter source failures")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        with pytest.raises(GuiRpcError) as rejected:
            tab.read_parameter("$qubit.missing", missing_ok=missing_ok)
        assert rejected.value.reason == expected
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
    finally:
        client.context.session.close()


def test_keep_defaults_are_omittable_but_null_reaches_the_form(tmp_path: Path) -> None:
    class DraftInputs(api.RecipeInputs):
        gain: float | None | api.Keep = Field(default=api.KEEP, description="Gain")

    assert DraftInputs().gain is api.KEEP
    assert DraftInputs.model_validate({"gain": None}).gain is None
    schema = DraftInputs.model_json_schema()
    assert "gain" not in schema.get("required", [])
    assert schema["properties"]["gain"]["anyOf"] == [
        {"type": "number"},
        {"type": "null"},
    ]
    assert "default" not in schema["properties"]["gain"]
    with pytest.raises(ValidationError):
        DraftInputs.model_validate({"gain": "KEEP"})
    with pytest.raises(ValidationError):
        DraftInputs.model_validate({"gain": float("inf")})

    @api.recipe("synthetic", "Keep omitted fields and apply ordered overrides last")
    async def synthetic(session: api.RecipeSession, inputs: DraftInputs) -> Report:
        tab = session.open_tab(
            "synthetic", target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id
        )
        tab.set("gain", inputs.gain)
        tab.select("pulse", api.KEEP)
        for edit in inputs.overrides:
            tab.set(edit.path, edit.value)
        return Report(count=1)

    client = make_client(tmp_path, SyntheticGui().respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context,
            "synthetic",
            {
                "overrides": [
                    {"path": "gain", "value": 0.3},
                    {"path": "gain", "value": None},
                ],
            },
        )
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        assert [
            p["edits"]
            for m, p in client.transport.sent
            if m == "tab.edit_recipe_fields"
        ] == [
            [{"path": ["gain"], "value": 0.3}],
            [{"path": ["gain"], "value": None}],
        ]
        assert next(p for m, p in client.transport.sent if m == "tab.reset_cfg") == {
            "tab_id": "tab-1"
        }
    finally:
        client.context.session.close()


def test_set_sends_null_and_original_index_while_accepting_incomplete_drafts(
    tmp_path: Path,
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        reply = gui.respond(method, params)
        if method == "tab.edit_recipe_fields":
            return {
                **reply,
                "errors": [{"path": ["other"], "message": "Field required"}],
            }
        return reply

    @api.recipe("synthetic", "Edit a draft before it is ready")
    async def synthetic(session: api.RecipeSession, inputs: Inputs) -> Report:
        tab = session.open_tab("synthetic")
        tab.set("tones.2.sweep.start", 100.0)
        tab.set("nullable", None)
        return Report(count=inputs.count)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = start(client, "synthetic")
        execution.join()
        assert execution.snapshot().status == "completed", execution.snapshot()
        assert [
            p["edits"]
            for m, p in client.transport.sent
            if m == "tab.edit_recipe_fields"
        ] == [
            [{"path": ["tones", 2, "sweep", "start"], "value": 100.0}],
            [{"path": ["nullable"], "value": None}],
        ]
        assert gui.runs == 0
    finally:
        client.context.session.close()
