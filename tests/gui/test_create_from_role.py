"""Controller.create_from_role: seed a blank ml entry from a named role,
md-linked defaults lowered to the md's current values.

Uses a real ExpContext (real MetaDict/ModuleLibrary) + a real RoleCatalog so the
factory → lowering → ml-register chain is exercised end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2_gui.registry import register_all_roles
from zcu_tools.gui.app.main.adapter import ContextReadiness, ExpContext
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.role_catalog import RoleCatalog, RoleEntry
from zcu_tools.gui.app.main.specs import make_pulse_spec
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.cfg import (
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.services.io_manager import IOManager
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctrl(
    md_values: dict,
    *,
    catalog: RoleCatalog | None = None,
) -> Controller:
    md = MetaDict()
    for k, v in md_values.items():
        setattr(md, k, v)
    ctx = ExpContext(
        md=md,
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        readiness=ContextReadiness.ACTIVE,
    )
    if catalog is None:
        catalog = RoleCatalog()
        register_all_roles(catalog)
    io = IOManager()
    io._em = MagicMock()  # simulate a project being set up
    bus = EventBus()
    return Controller(
        state=State(ctx),
        registry=Registry(),
        io_manager=io,
        view=None,
        bus=bus,
        role_catalog=catalog,
    )


def _instrumented_entry(
    events: list[str],
    *,
    fail_value: bool = False,
    fail_shape_on_create: bool = False,
) -> tuple[RoleEntry, list[object], list[object]]:
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
        RoleEntry("instrumented", "Instrumented", "module", shape, make_value),
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


def test_create_module_from_role_uses_md_value(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({"r_f": 6123.0, "res_ch": 1, "ro_ch": 2})
    ctrl.create_from_role("module", "res_probe", "my_ro")

    ml = ctrl.get_current_ml()
    assert "my_ro" in ml.modules
    raw = ml.modules["my_ro"].to_dict()
    # res_probe is a bare pulse: md-linked freq lowered to the md's current
    # value (not a structural 0.0).
    assert raw["freq"] == 6123.0


def test_create_from_role_uses_value_then_fresh_shape_exactly_once(qapp) -> None:  # noqa: ARG001
    events: list[str] = []
    entry, made_specs, made_values = _instrumented_entry(events)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()
    made_specs.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    ctrl.set_ml_module_from_schema = MagicMock()  # type: ignore[method-assign]

    ctrl.create_from_role("module", "instrumented", "created")

    assert events == ["value", "shape"]
    assert len(made_specs) == 1
    assert len(made_values) == 1
    assert get_context.call_count == 1
    schema = ctrl.set_ml_module_from_schema.call_args.args[1]
    assert schema.spec is made_specs[0]
    assert schema.value is made_values[0]


def test_create_from_role_value_failure_does_not_call_shape(qapp) -> None:  # noqa: ARG001
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events, fail_value=True)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()

    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    ctrl.set_ml_module_from_schema = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="value failed"):
        ctrl.create_from_role("module", "instrumented", "created")

    assert events == ["value"]
    assert get_context.call_count == 1
    ctrl.set_ml_module_from_schema.assert_not_called()


def test_create_from_role_shape_failure_occurs_after_value(qapp) -> None:  # noqa: ARG001
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events, fail_shape_on_create=True)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()

    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    ctrl.set_ml_module_from_schema = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="shape failed"):
        ctrl.create_from_role("module", "instrumented", "created")

    assert events == ["value", "shape"]
    assert get_context.call_count == 1
    ctrl.set_ml_module_from_schema.assert_not_called()


def test_create_from_role_context_failure_calls_no_factory_or_write(qapp) -> None:  # noqa: ARG001
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    ctrl.get_exp_context = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("context failed")
    )
    ctrl.set_ml_module_from_schema = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="context failed"):
        ctrl.create_from_role("module", "instrumented", "created")

    assert events == []
    assert ctrl.get_exp_context.call_count == 1
    ctrl.set_ml_module_from_schema.assert_not_called()


def test_create_from_role_downstream_failure_preserves_factory_counts_and_identity(
    qapp,  # noqa: ARG001
) -> None:
    events: list[str] = []
    entry, made_specs, made_values = _instrumented_entry(events)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()
    made_specs.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    ctrl.set_ml_module_from_schema = MagicMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("write failed")
    )

    with pytest.raises(RuntimeError, match="write failed"):
        ctrl.create_from_role("module", "instrumented", "created")

    assert events == ["value", "shape"]
    assert get_context.call_count == 1
    assert ctrl.set_ml_module_from_schema.call_count == 1
    schema = ctrl.set_ml_module_from_schema.call_args.args[1]
    assert schema.spec is made_specs[0]
    assert schema.value is made_values[0]


@pytest.mark.parametrize(
    ("item_kind", "name", "error"),
    [
        ("module", "", "name must not be empty"),
        ("waveform", "created", "not a waveform"),
    ],
)
def test_create_from_role_guards_do_not_call_value_or_shape(
    qapp,  # noqa: ARG001
    item_kind: str,
    name: str,
    error: str,
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    ctrl.set_ml_module_from_schema = MagicMock()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match=error):
        ctrl.create_from_role(item_kind, "instrumented", name)

    assert events == []
    get_context.assert_not_called()
    ctrl.set_ml_module_from_schema.assert_not_called()


def test_create_from_role_name_clash_guard_does_not_call_value_or_shape(qapp) -> None:  # noqa: ARG001
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = RoleCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl({}, catalog=catalog)
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    ctrl.set_ml_module_from_schema = MagicMock()  # type: ignore[method-assign]
    ctrl.get_current_ml().register_module(existing=_pulse_raw())

    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_role("module", "instrumented", "existing")

    assert events == []
    get_context.assert_not_called()
    ctrl.set_ml_module_from_schema.assert_not_called()


def test_create_module_from_role_empty_md_falls_back(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    ctrl.create_from_role("module", "res_probe", "ro_blank")

    raw = ctrl.get_current_ml().modules["ro_blank"].to_dict()
    # fallback literal (the factory's default), not a crash.
    assert raw["freq"] == 6000.0


def test_create_from_role_name_clash_fails(qapp):  # noqa: ARG001
    """Create is new-entry semantics: a name clash must fail fast, not silently
    overwrite an existing ml entry."""
    ctrl = _make_ctrl({"r_f": 6000.0})
    ctrl.create_from_role("module", "res_probe", "dup")
    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_role("module", "qub_probe", "dup")
    # the original entry is untouched (not overwritten by the failed second call)
    assert ctrl.get_current_ml().modules["dup"].to_dict()["type"] == "pulse"


def test_create_waveform_from_role(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    ctrl.create_from_role("waveform", "res_waveform", "ro_wav")
    assert "ro_wav" in ctrl.get_current_ml().waveforms


def test_create_from_blank_module_role(qapp):  # noqa: ARG001
    """A ':blank' role creates a structural-zero entry of that exact shape."""
    ctrl = _make_ctrl({"r_f": 6000.0})
    ctrl.create_from_role("module", "reset/bath:blank", "rb")
    raw = ctrl.get_current_ml().modules["rb"].to_dict()
    assert raw["type"] == "reset/bath"


def test_create_from_blank_waveform_role_uncovered_style(qapp):  # noqa: ARG001
    """A waveform style with no md-aware role (drag) is reachable via :blank."""
    ctrl = _make_ctrl({})
    ctrl.create_from_role("waveform", "drag:blank", "dwav")
    raw = ctrl.get_current_ml().waveforms["dwav"].to_dict()
    assert raw["style"] == "drag"


def test_item_kind_mismatch_raises(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    with pytest.raises(RuntimeError, match="not a waveform"):
        ctrl.create_from_role("waveform", "res_probe", "x")


def test_unknown_role_raises(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    get_context = MagicMock(wraps=ctrl.get_exp_context)
    ctrl.get_exp_context = get_context  # type: ignore[method-assign]
    with pytest.raises(KeyError):
        ctrl.create_from_role("module", "no_such_role", "x")
    get_context.assert_not_called()


def test_empty_name_raises(qapp):  # noqa: ARG001
    ctrl = _make_ctrl({})
    with pytest.raises(RuntimeError, match="name must not be empty"):
        ctrl.create_from_role("module", "res_probe", "")


def test_no_catalog_wired_raises(qapp):  # noqa: ARG001
    ctx = ExpContext(
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
    with pytest.raises(RuntimeError, match="No role catalog"):
        ctrl.create_from_role("module", "res_probe", "x")
