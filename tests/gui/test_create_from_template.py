"""Controller.create_from_template: seed a blank ml entry from a named role,
md-linked defaults lowered to the md's current values.

Uses a real SessionEnv (real MetaDict/ModuleLibrary) + a real TemplateCatalog so the
factory → lowering → ml-register chain is exercised end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.adapter import ContextReadiness, SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.specs import make_pulse_spec
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog, TemplateEntry
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from zcu_lab.templates import register_all_templates


def _make_ctrl(
    md_values: dict,
    *,
    catalog: TemplateCatalog | None = None,
) -> Controller:
    md = MetaDict()
    for k, v in md_values.items():
        setattr(md, k, v)
    ctx = SessionEnv(
        md=md,
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    if catalog is None:
        catalog = TemplateCatalog()
        register_all_templates(catalog)
    io = IOManager()
    io._em = MagicMock()  # simulate a project being set up
    bus = EventBus()
    return Controller(
        state=State(ctx),
        registry=Registry(),
        io_manager=io,
        view=None,
        bus=bus,
        template_catalog=catalog,
    )


def _instrumented_entry(
    events: list[str],
    *,
    fail_value: bool = False,
    fail_shape_on_create: bool = False,
) -> tuple[TemplateEntry, list[object], list[object]]:
    made_specs: list[object] = []
    made_values: list[object] = []
    shape_calls = 0

    def shape():
        nonlocal shape_calls
        shape_calls += 1
        events.append("shape")
        if fail_shape_on_create and shape_calls > 1:
            raise RuntimeError("shape failed")
        spec = make_pulse_spec()
        made_specs.append(spec)
        return spec

    def make_value(_ctx):
        events.append("value")
        if fail_value:
            raise RuntimeError("value failed")
        spec = make_pulse_spec()
        ref = ReferenceValue(
            make_custom_reference_key("pulse"),
            make_default_value(spec),
        )
        made_values.append(ref.value)
        return ref

    return (
        TemplateEntry("instrumented", "Instrumented", "module", shape, make_value),
        made_specs,
        made_values,
    )


def _pulse_raw() -> dict[str, object]:
    return {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 0.0,
        "gain": 0.0,
        "phase": 0.0,
        "pre_delay": 0.0,
        "post_delay": 0.0,
        "waveform": {"style": "const", "length": 0.0},
    }


def test_create_module_from_role_uses_md_value(qapp):
    ctrl = _make_ctrl({"r_f": 6123.0, "res_ch": 1, "ro_ch": 2})
    ctrl.create_from_template("module", "res_probe", "my_ro")

    ml = ctrl.get_current_ml()
    assert "my_ro" in ml.modules
    raw = ml.modules["my_ro"].to_dict()
    # res_probe is a bare pulse: md-linked freq lowered to the md's current
    # value (not a structural 0.0).
    assert raw["freq"] == 6123.0


def test_create_from_template_uses_value_then_fresh_shape_exactly_once(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    entry, made_specs, made_values = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    made_specs.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock()
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)

    ctrl.create_from_template("module", "instrumented", "created")

    assert events == ["value", "shape"]
    assert len(made_specs) == 1
    assert len(made_values) == 1
    assert get_context.call_count == 1
    schema = write.call_args.args[1]
    assert schema.spec is made_specs[0]
    assert schema.value is made_values[0]


def test_create_from_template_value_failure_does_not_call_shape(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events, fail_value=True)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()

    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock()
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)

    with pytest.raises(RuntimeError, match="value failed"):
        ctrl.create_from_template("module", "instrumented", "created")

    assert events == ["value"]
    assert get_context.call_count == 1
    write.assert_not_called()


def test_create_from_template_shape_failure_occurs_after_value(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events, fail_shape_on_create=True)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()

    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock()
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)

    with pytest.raises(RuntimeError, match="shape failed"):
        ctrl.create_from_template("module", "instrumented", "created")

    assert events == ["value", "shape"]
    assert get_context.call_count == 1
    write.assert_not_called()


def test_create_from_template_context_failure_calls_no_factory_or_write(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(side_effect=RuntimeError("context failed"))
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock()
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)

    with pytest.raises(RuntimeError, match="context failed"):
        ctrl.create_from_template("module", "instrumented", "created")

    assert events == []
    assert get_context.call_count == 1
    write.assert_not_called()


def test_create_from_template_downstream_failure_preserves_factory_counts_and_identity(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    entry, made_specs, made_values = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    made_specs.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock(side_effect=RuntimeError("write failed"))
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)

    with pytest.raises(RuntimeError, match="write failed"):
        ctrl.create_from_template("module", "instrumented", "created")

    assert events == ["value", "shape"]
    assert get_context.call_count == 1
    assert write.call_count == 1
    schema = write.call_args.args[1]
    assert schema.spec is made_specs[0]
    assert schema.value is made_values[0]


@pytest.mark.parametrize(
    ("item_kind", "name", "error"),
    [
        ("module", "", "name must not be empty"),
        ("waveform", "created", "not a waveform"),
    ],
)
def test_create_from_template_guards_do_not_call_value_or_shape(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    item_kind: str,
    name: str,
    error: str,
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock()
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)

    with pytest.raises(RuntimeError, match=error):
        ctrl.create_from_template(item_kind, "instrumented", name)

    assert events == []
    get_context.assert_not_called()
    write.assert_not_called()


def test_create_from_template_name_clash_guard_does_not_call_value_or_shape(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    write = MagicMock()
    monkeypatch.setattr(ctrl, "set_ml_module_from_schema", write)
    ctrl.get_current_ml().register_module(existing=_pulse_raw())

    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_template("module", "instrumented", "existing")

    assert events == []
    get_context.assert_not_called()
    write.assert_not_called()


def test_create_module_from_role_empty_md_falls_back(qapp):
    ctrl = _make_ctrl({})
    ctrl.create_from_template("module", "res_probe", "ro_blank")

    raw = ctrl.get_current_ml().modules["ro_blank"].to_dict()
    # fallback literal (the factory's default), not a crash.
    assert raw["freq"] == 6000.0


def test_create_from_template_name_clash_fails(qapp):
    """Create is new-entry semantics: a name clash must fail fast, not silently
    overwrite an existing ml entry."""
    ctrl = _make_ctrl({"r_f": 6000.0})
    ctrl.create_from_template("module", "res_probe", "dup")
    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_template("module", "qub_probe", "dup")
    # the original entry is untouched (not overwritten by the failed second call)
    assert ctrl.get_current_ml().modules["dup"].to_dict()["type"] == "pulse"


def test_create_waveform_from_role(qapp):
    ctrl = _make_ctrl({})
    ctrl.create_from_template("waveform", "res_waveform", "ro_wav")
    assert "ro_wav" in ctrl.get_current_ml().waveforms


def test_create_from_blank_module_role(qapp):
    """A ':blank' role creates a structural-zero entry of that exact shape."""
    ctrl = _make_ctrl({"r_f": 6000.0})
    ctrl.create_from_template("module", "reset/bath:blank", "rb")
    raw = ctrl.get_current_ml().modules["rb"].to_dict()
    assert raw["type"] == "reset/bath"


def test_create_from_blank_waveform_role_uncovered_style(qapp):
    """A waveform style with no md-aware role (drag) is reachable via :blank."""
    ctrl = _make_ctrl({})
    ctrl.create_from_template("waveform", "drag:blank", "dwav")
    raw = ctrl.get_current_ml().waveforms["dwav"].to_dict()
    assert raw["style"] == "drag"


def test_item_kind_mismatch_raises(qapp):
    ctrl = _make_ctrl({})
    with pytest.raises(RuntimeError, match="not a waveform"):
        ctrl.create_from_template("waveform", "res_probe", "x")


def test_unknown_role_raises(qapp, monkeypatch: pytest.MonkeyPatch):
    ctrl = _make_ctrl({})
    get_context = MagicMock(wraps=ctrl.get_session_env)
    monkeypatch.setattr(ctrl, "get_session_env", get_context)
    with pytest.raises(KeyError):
        ctrl.create_from_template("module", "no_such_role", "x")
    get_context.assert_not_called()


def test_empty_name_raises(qapp):
    ctrl = _make_ctrl({})
    with pytest.raises(RuntimeError, match="name must not be empty"):
        ctrl.create_from_template("module", "res_probe", "")


def test_no_catalog_wired_raises(qapp):
    ctx = SessionEnv(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    io = IOManager()
    io._em = MagicMock()
    ctrl = Controller(
        state=State(ctx),
        registry=Registry(),
        io_manager=io,
        view=None,
        bus=EventBus(),
    )
    with pytest.raises(RuntimeError, match="No template catalog"):
        ctrl.create_from_template("module", "res_probe", "x")
