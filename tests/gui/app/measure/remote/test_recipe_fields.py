"""Logical recipe RPCs retain owner guards, atomic edits and captured sources."""

import pytest

from ._form_rpc import ContainerRpcAdapter, ModuleRpcAdapter


def test_recipe_rpc_logical_batch_invalid_draft_and_atomic_rejection(form_rpc) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    base = owner.editor_state.base.data
    version_key = f"tab:{tab}:cfg"
    version = fixture.state.version.get(version_key)
    notifications: list[str] = []
    owner.subscribe(lambda: notifications.append("changed"))
    changed = rpc.call(
        "tab.edit_recipe_fields",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": 0.75},
                {"path": ["sweep", "start"], "value": 8.0},
            ],
        },
    )
    assert changed["ok"], changed
    assert changed["result"] == {"changed": True, "errors": []}
    assert notifications == ["changed"]
    assert fixture.state.version.get(version_key) == version + 1
    assert owner.editor_state.base.data == base
    before = owner.input.data
    rejected = rpc.call(
        "tab.edit_recipe_fields",
        {
            "tab_id": tab,
            "edits": [
                {"path": ["gain"], "value": 0.1},
                {"path": ["missing"], "value": 1},
            ],
        },
    )
    assert not rejected["ok"], rejected
    assert rejected["error"]["reason"] == "form_path"
    assert rejected["error"]["data"]["form_path"] == ["missing"]
    assert rejected["error"]["data"]["edit_index"] == 1
    assert owner.input.data == before
    assert notifications == ["changed"]
    invalid = rpc.call(
        "tab.edit_recipe_fields",
        {
            "tab_id": tab,
            "edits": [{"path": ["gain"], "value": {"expr": "1 / 0"}}],
        },
    )
    assert invalid["ok"], invalid
    assert invalid["result"]["changed"] is True
    assert invalid["result"]["errors"][0]["path"] == ["gain"]
    assert owner.input.data["gain"] == {"expr": "1 / 0"}


@pytest.mark.parametrize(
    "selection",
    [
        {"action": "use_library"},
        {"action": "use_library", "key": None},
        {"action": "use_library", "key": " "},
        {"action": "use_inline", "key": None},
        {"action": "disable", "key": "unexpected"},
    ],
)
def test_recipe_selector_rejects_bad_key_shape_without_effects(
    form_rpc, selection
) -> None:
    fixture, tab, rpc = form_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    before = owner.input.data
    result = rpc.call(
        "tab.select_recipe_module",
        {
            "tab_id": tab,
            "path": ["pulse"],
            **selection,
        },
    )
    assert not result["ok"], result
    assert result["error"]["code"] == "invalid_params"
    assert result["error"]["reason"] == "form_errors"
    assert owner.input.data == before


def test_recipe_rpc_read_and_noop_preserve_observer_busy_and_notification_policy(
    form_rpc,
) -> None:
    fixture, _, rpc = form_rpc
    fixture.registry.register("logical-module", ModuleRpcAdapter)
    tab = fixture.ctrl.new_tab("logical-module")
    try:
        owner = fixture.ctrl.cfg_resources.lookup(tab)
        owner.write({("pulse",): {"kind": "inline"}})
        version_key = f"tab:{tab}:cfg"
        version = fixture.state.version.get(version_key)
        selection = {"tab_id": tab, "path": ["pulse"], "action": "use_inline"}
        noop = rpc.call("tab.select_recipe_module", selection)
        assert noop["ok"], noop
        assert noop["result"]["changed"] is False
        assert noop["result"]["field"]["selector"]["mode"] == "inline"
        assert fixture.state.version.get(version_key) == version
        fixture.state.set_tab_running(tab, running=True)
        busy = rpc.call("tab.select_recipe_module", selection)
        assert not busy["ok"]
        assert busy["error"]["reason"] == "mutation_blocked"
        fixture.ctrl.control_identity.reclaim_local()
        read = rpc.call("tab.read_recipe_field", {"tab_id": tab, "path": ["pulse"]})
        assert read["ok"], read
        assert read["result"]["selector"]["mode"] == "inline"
        denied = rpc.call("tab.select_recipe_module", selection)
        assert not denied["ok"]
        assert denied["error"]["reason"] == "not_holder"
    finally:
        fixture.state.set_tab_running(tab, running=False)
        fixture.ctrl.tab_control.close_tab(tab)


@pytest.fixture
def nested_recipe_rpc(form_rpc, tmp_path):
    from contextlib import closing

    from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
    from zcu_tools.gui.remote.errors import RemoteError

    from tests.mcp.measure._support import make_client
    from tests.resources.entry.fakes import (
        register_fakes,
        registry_state,
        restore_registry,
    )

    fixture, _, rpc = form_rpc
    registry_before = registry_state()
    tab = None
    try:
        register_fakes()
        context = fixture.ctrl.context_control
        context.add_component("Q1", "fake/drive/a", {})
        target = context.read_context().target
        context.set_waveform(target, "Q1.waveforms.seed", ConstWaveformCfg(length=0.1))
        context.set_module(
            target,
            "Q1.pulses.pi",
            PulseSpec(
                ch=0,
                nqz=1,
                freq=100.0,
                gain=0.2,
                waveform="Q1.waveforms.seed",
            ),
        )
        fixture.registry.register("nested-recipe", ModuleRpcAdapter)
        tab = fixture.ctrl.new_tab("nested-recipe")

        def respond(method, params):
            reply = rpc.call(method, params)
            if not reply["ok"]:
                error = reply["error"]
                raise RemoteError(
                    error["code"],
                    error["message"],
                    reason=error.get("reason"),
                    data=error.get("data"),
                )
            return reply["result"]

        client = make_client(tmp_path, respond)
        with closing(client.context.session):
            yield fixture, tab, client, context, target
    finally:
        if tab is not None and tab in fixture.ctrl.list_tab_ids():
            fixture.ctrl.tab_control.close_tab(tab)
        restore_registry(registry_before)


@pytest.fixture
def container_recipe_rpc(nested_recipe_rpc):
    fixture, _, client, _, _ = nested_recipe_rpc
    fixture.registry.register("container-recipe", ContainerRpcAdapter)
    tab = fixture.ctrl.new_tab("container-recipe")
    try:
        yield fixture, tab, client
    finally:
        fixture.ctrl.tab_control.close_tab(tab)


@pytest.mark.parametrize("value", [{}, 4])
def test_recipe_rpc_structure_rejection_identifies_first_edit_atomically(
    container_recipe_rpc, value
) -> None:
    from zcu_tools.mcp.measure.session import GuiRpcError

    fixture, tab, client = container_recipe_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    before = owner.input.to_json()
    base = owner.editor_state.base.to_json()
    version_key = f"tab:{tab}:cfg"
    version = fixture.state.version.get(version_key)
    notices: list[str] = []
    owner.subscribe(lambda: notices.append("changed"))
    with pytest.raises(GuiRpcError) as rejected:
        client.context.session.send_gui_rpc(
            "tab.edit_recipe_fields",
            {
                "tab_id": tab,
                "edits": [
                    {"path": ["extras", "real"], "value": value},
                    {"path": ["gain"], "value": 0.9},
                ],
            },
        )
    assert rejected.value.code == "invalid_params"
    assert rejected.value.reason == "form_structure"
    assert rejected.value.data is not None
    assert rejected.value.data["path"] == ["extras", "real"]
    assert rejected.value.data["edit_index"] == 0
    assert owner.input.to_json() == before
    assert owner.editor_state.base.to_json() == base
    assert fixture.state.version.get(version_key) == version
    assert notices == []


def test_recipe_rpc_container_payload_cannot_replace_referenced_selector(
    container_recipe_rpc,
) -> None:
    from zcu_tools.mcp.measure.session import GuiRpcError

    fixture, tab, client = container_recipe_rpc
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.write(
        {("extras", "group"): {"drive": {"kind": "ref", "name": "Q1.pulses.pi"}}}
    )
    before = owner.input.to_json()
    base = owner.editor_state.base.to_json()
    version_key = f"tab:{tab}:cfg"
    version = fixture.state.version.get(version_key)
    notices: list[str] = []
    owner.subscribe(lambda: notices.append("changed"))
    with pytest.raises(GuiRpcError) as rejected:
        client.context.session.send_gui_rpc(
            "tab.edit_recipe_fields",
            {
                "tab_id": tab,
                "edits": [
                    {"path": ["gain"], "value": 0.9},
                    {
                        "path": ["extras", "group"],
                        "value": {
                            "drive": {
                                "kind": "inline",
                                "module": {
                                    "type": "pulse",
                                    "ch": 0,
                                    "nqz": 1,
                                    "freq": 999.0,
                                    "gain": 0.8,
                                    "waveform": {
                                        "kind": "ref",
                                        "name": "Q1.waveforms.seed",
                                    },
                                },
                            }
                        },
                    },
                ],
            },
        )
    assert rejected.value.code == "invalid_params"
    assert rejected.value.reason == "module_selector"
    assert rejected.value.data is not None
    assert rejected.value.data["path"] == ["extras", "group"]
    assert rejected.value.data["edit_index"] == 1
    assert owner.input.to_json() == before
    assert owner.editor_state.base.to_json() == base
    assert fixture.state.version.get(version_key) == version
    assert notices == []


def assert_run_keeps_nested_length_after_updates(prepared, nested_recipe_rpc) -> None:
    """Observe the original Run after successful library and draft mutations."""
    from zcu_tools.cfg_model import ConstWaveformCfg

    from tests.gui._qt_waits import qt_until

    fixture, tab, client, context, target = nested_recipe_rpc
    operation = prepared.run()
    # The accepted synthetic Run returns immediately and queues its owner delivery.
    qt_until(
        lambda: not fixture.state.is_tab_running(tab), label="synthetic recipe Run"
    )
    result = client.context.session.send_gui_rpc("tab.snapshot", {"tab_id": tab})
    assert result["tabs"][0]["result_state"]["available"] is True
    captured = operation.snapshot()
    assert captured is not None
    assert captured.actual["fields"]["pulse.waveform.length"]["value"] == 0.08
    context.set_waveform(target, "Q1.waveforms.seed", ConstWaveformCfg(length=0.7))
    prepared.set("pulse.waveform.length", 0.16)
    assert prepared.read_field("pulse.waveform.length").value == 0.16
    after_updates = operation.snapshot()
    assert after_updates is not None
    assert after_updates.actual["fields"]["pulse.waveform.length"]["value"] == 0.08


def test_recipe_tab_uses_real_rpc_for_nested_captured_refs_and_copy(
    nested_recipe_rpc,
) -> None:
    from zcu_tools.cfg_model import ConstWaveformCfg
    from zcu_tools.mcp.measure.execution_reply import SummaryParameter
    from zcu_tools.mcp.measure.recipe import RecipeFieldEdit, RecipeSession
    from zcu_tools.mcp.measure.session import GuiRpcError

    fixture, tab, client, context, target = nested_recipe_rpc
    prepared = RecipeSession(
        client.context,
        summary_parameters=(SummaryParameter("length", "pulse.waveform.length"),),
    ).open_tab("nested-recipe", reuse=tab)
    prepared.use_library("pulse", "Q1.pulses.pi")
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    before = owner.input.data
    base = owner.editor_state.base.data
    version = fixture.state.version.get(f"tab:{tab}:cfg")
    notices: list[str] = []
    owner.subscribe(lambda: notices.append("changed"))
    context.set_waveform(target, "Q1.waveforms.seed", ConstWaveformCfg(length=0.9))
    field = prepared.read_field("pulse.waveform.length")
    assert field.value == 0.1 and field.readonly
    assert field.source is not None
    assert field.source["key"] == "Q1.waveforms.seed"
    assert field.source["selector_path"] == ["pulse", "waveform"]
    assert field.input_present is False and field.mode is None
    assert owner.input.data == before
    assert owner.editor_state.base.data == base
    assert fixture.state.version.get(f"tab:{tab}:cfg") == version
    assert notices == []
    with pytest.raises(GuiRpcError) as rejected:
        prepared.set_fields(
            [RecipeFieldEdit("gain", 0.9), RecipeFieldEdit("pulse.freq", 125.0)]
        )
    assert rejected.value.reason == "ref_ancestor"
    data = rejected.value.data
    assert data is not None
    assert data["edit_index"] == 1
    assert owner.input.data == before
    assert notices == []
    with pytest.raises(GuiRpcError) as ancestor:
        prepared.use_inline("pulse.waveform")
    assert ancestor.value.reason == "ref_ancestor"
    prepared.use_inline("pulse")
    prepared.use_inline("pulse.waveform")
    prepared.set("pulse.waveform.length", 0.08)
    prepared.retain_source("pulse.waveform.length", "explicit")
    assert prepared.read_field("pulse.waveform.length").value == 0.08
    assert_run_keeps_nested_length_after_updates(prepared, nested_recipe_rpc)
    assert field.value == 0.1
    fixture.ctrl.close()


@pytest.mark.parametrize("operation", ["read", "copy"])
def test_recipe_source_failure_preserves_identity_and_draft(
    nested_recipe_rpc, operation
) -> None:
    from zcu_tools.cfg_model import PulseSpec
    from zcu_tools.mcp.measure.session import GuiRpcError

    fixture, tab, client, context, target = nested_recipe_rpc
    context.set_module(
        target,
        "Q1.pulses.pi",
        PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.missing"),
    )
    owner = fixture.ctrl.cfg_resources.lookup(tab)
    owner.reset()
    # The source exists at the selected module, but its captured child key does not.
    owner.select_recipe_module(("pulse",), "use_library", key="Q1.pulses.pi")
    if operation == "copy":
        owner.select_recipe_module(("pulse",), "use_inline")
    before = owner.input.data
    base = owner.editor_state.base.data
    version = fixture.state.version.get(f"tab:{tab}:cfg")
    notices: list[str] = []
    owner.subscribe(lambda: notices.append("changed"))
    path = (
        ["pulse", "waveform", "length"]
        if operation == "read"
        else ["pulse", "waveform"]
    )
    method = (
        "tab.read_recipe_field" if operation == "read" else "tab.select_recipe_module"
    )
    params = {"tab_id": tab, "path": path}
    if operation == "copy":
        params["action"] = "use_inline"
    # Use the same in-process RPC transport as RecipeTab, retaining its native error.
    with pytest.raises(GuiRpcError) as rejected:
        client.context.session.send_gui_rpc(method, params)
    assert rejected.value.code == "precondition_failed"
    assert rejected.value.reason == "library_source"
    assert rejected.value.data == {
        "path": path,
        "selector_path": ["pulse", "waveform"],
        "key": "Q1.waveforms.missing",
        "source_kind": "waveform",
    }
    assert owner.input.data == before
    assert owner.editor_state.base.data == base
    assert fixture.state.version.get(f"tab:{tab}:cfg") == version
    assert notices == []
