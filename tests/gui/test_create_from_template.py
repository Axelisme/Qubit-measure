"""Create templates through the real entry, parameter and library owners."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from zcu_tools.cfg_model import module_spec_from_raw
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.specs import make_pulse_spec
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog, TemplateEntry
from zcu_tools.gui.cfg import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.events import MlChangedPayload
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import component_registry

from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.templates import register_all_templates


def _make_ctrl(
    root: Path,
    resonator_fields: YamlMap | None = None,
    *,
    catalog: TemplateCatalog | None = None,
) -> Controller:
    if catalog is None:
        catalog = TemplateCatalog()
        register_all_templates(catalog)
    ctrl = Controller(
        state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry=Registry(),
        view=None,
        bus=EventBus(),
        template_catalog=catalog,
        project_root=str(root),
    )
    ctrl.setup_control.new_entry(EntryRequest("templates"))
    before = component_registry.kinds()
    added: list[str] = []
    try:
        for kind, model in [
            ("qubit/transmon", QubitSchema),
            ("resonator", ResonatorSchema),
        ]:
            if kind not in before:
                component_registry.register(kind, model)
                added.append(kind)
            assert component_registry.get(kind) is model
        ctrl.context_control.add_component("Q1", "qubit/transmon", {}, setup=True)
        ctrl.context_control.add_component(
            "R1", "resonator", resonator_fields or {}, setup=True
        )
        ctrl.setup_control.new_point(
            PointCreateRequest("working", "", CommandOrigin("gui"))
        )
    finally:
        for kind in reversed(added):
            component_registry.unregister(kind)
        assert component_registry.kinds() == before
    ctrl.context_control.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "R1"))
    )
    return ctrl


def _instrumented_entry(
    events: list[str],
    *,
    fail_value: bool = False,
    fail_shape_on_create: bool = False,
) -> tuple[
    TemplateEntry,
    list[tuple[CfgSectionSpec, CfgSectionSpec]],
    list[tuple[CfgSectionValue, CfgSectionValue]],
]:
    made_specs: list[tuple[CfgSectionSpec, CfgSectionSpec]] = []
    made_values: list[tuple[CfgSectionValue, CfgSectionValue]] = []
    shape_calls = 0

    def shape() -> CfgSectionSpec:
        nonlocal shape_calls
        shape_calls += 1
        events.append("shape")
        if fail_shape_on_create and shape_calls > 1:
            raise RuntimeError("shape failed")
        spec = make_pulse_spec()
        made_specs.append((spec, deepcopy(spec)))
        return spec

    def make_value(_ctx: SessionEnv) -> ReferenceValue:
        events.append("value")
        if fail_value:
            raise RuntimeError("value failed")
        ref = ReferenceValue(
            make_custom_reference_key("pulse"),
            make_default_value(make_pulse_spec()),
        )
        assert ref.value is not None
        made_values.append((ref.value, deepcopy(ref.value)))
        return ref

    return (
        TemplateEntry("instrumented", "Instrumented", "module", shape, make_value),
        made_specs,
        made_values,
    )


def _pulse_raw() -> YamlMap:
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


def _library_keys(ctrl: Controller) -> tuple[tuple[str, ...], tuple[str, ...]]:
    library = ctrl.read_context().library
    return library.module_keys(), library.waveform_keys()


def test_create_module_from_role_uses_md_value(qapp, tmp_path: Path):
    ctrl = _make_ctrl(tmp_path, {"freq": 6123.0, "wiring": {"ch": 1, "ro_ch": 2}})
    ctrl.create_from_template("module", "res_probe", "R1.pulses.my_ro")

    library = ctrl.read_context().library
    assert "R1.pulses.my_ro" in library.module_keys()
    raw = library.get_module("R1.pulses.my_ro").to_dict()
    assert raw["freq"] == 6123.0


def test_create_from_template_uses_value_then_fresh_shape_exactly_once(
    qapp, tmp_path: Path
) -> None:
    events: list[str] = []
    entry, made_specs, made_values = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    made_specs.clear()
    ctrl = _make_ctrl(tmp_path, catalog=catalog)

    ctrl.create_from_template("module", "instrumented", "Q1.pulses.created")

    assert events == ["value", "shape"]
    assert len(made_specs) == 1
    assert len(made_values) == 1
    library = ctrl.read_context().library
    raw = library.get_module("Q1.pulses.created").to_dict()
    assert raw["freq"] == 0.0
    assert raw["gain"] == 0.0
    assert raw["ch"] == 0
    assert library.module_spec("Q1.pulses.created").type == "pulse"
    assert made_values[0][0].fields["freq"] == DirectValue(raw["freq"])
    assert made_specs[0][0] == made_specs[0][1]
    assert made_values[0][0] == made_values[0][1]


def test_create_from_template_value_failure_does_not_call_shape(
    qapp, tmp_path: Path
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events, fail_value=True)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl(tmp_path, catalog=catalog)
    before = _library_keys(ctrl)

    with pytest.raises(RuntimeError, match="value failed"):
        ctrl.create_from_template("module", "instrumented", "Q1.pulses.created")

    assert events == ["value"]
    assert _library_keys(ctrl) == before


def test_create_from_template_shape_failure_occurs_after_value(
    qapp, tmp_path: Path
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events, fail_shape_on_create=True)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl(tmp_path, catalog=catalog)
    before = _library_keys(ctrl)

    with pytest.raises(RuntimeError, match="shape failed"):
        ctrl.create_from_template("module", "instrumented", "Q1.pulses.created")

    assert events == ["value", "shape"]
    assert _library_keys(ctrl) == before


def test_create_from_template_context_failure_calls_no_factory_or_write(qapp) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    bus = EventBus()
    writes: list[MlChangedPayload] = []
    bus.subscribe(MlChangedPayload, writes.append)
    ctrl = Controller(
        state=State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry=Registry(),
        view=None,
        bus=bus,
        template_catalog=catalog,
    )

    with pytest.raises(FailedPreconditionError) as caught:
        ctrl.create_from_template("module", "instrumented", "Q1.pulses.created")

    assert caught.value.reason_code == "no_context"
    assert events == []
    assert writes == []


def test_create_from_template_downstream_failure_preserves_factory_counts_and_identity(
    qapp, tmp_path: Path
) -> None:
    events: list[str] = []
    entry, made_specs, made_values = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    made_specs.clear()
    ctrl = _make_ctrl(tmp_path, catalog=catalog)
    before = _library_keys(ctrl)
    info = ctrl.setup_control.get_entry_info()
    assert info is not None
    (info.result_path / "points" / "working" / "module_cfg.yaml").unlink()

    with pytest.raises(FileNotFoundError):
        ctrl.create_from_template("module", "instrumented", "Q1.pulses.created")

    assert events == ["value", "shape"]
    assert len(made_specs) == 1
    assert len(made_values) == 1
    assert made_specs[0][0] == made_specs[0][1]
    assert made_values[0][0] == made_values[0][1]
    assert _library_keys(ctrl) == before


@pytest.mark.parametrize(
    ("item_kind", "name", "error"),
    [
        ("module", "", "name must not be empty"),
        ("waveform", "created", "not a waveform"),
    ],
)
def test_create_from_template_guards_do_not_call_value_or_shape(
    qapp,
    tmp_path: Path,
    item_kind: str,
    name: str,
    error: str,
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl(tmp_path, catalog=catalog)
    before = _library_keys(ctrl)

    with pytest.raises(RuntimeError, match=error):
        ctrl.create_from_template(item_kind, "instrumented", name)

    assert events == []
    assert _library_keys(ctrl) == before


def test_create_from_template_name_clash_guard_does_not_call_value_or_shape(
    qapp, tmp_path: Path
) -> None:
    events: list[str] = []
    entry, _, _ = _instrumented_entry(events)
    catalog = TemplateCatalog()
    catalog.register(entry)
    events.clear()
    ctrl = _make_ctrl(tmp_path, catalog=catalog)
    ctrl.context_control.set_module(
        ctrl.read_context().target,
        "Q1.pulses.existing",
        module_spec_from_raw(_pulse_raw()),
    )
    before = _library_keys(ctrl)

    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_template("module", "instrumented", "Q1.pulses.existing")

    assert events == []
    assert _library_keys(ctrl) == before


def test_create_module_from_role_without_calibration_fails_without_writing(
    qapp, tmp_path: Path
):
    ctrl = _make_ctrl(tmp_path)
    before = _library_keys(ctrl)
    with pytest.raises(InvalidInputError) as caught:
        ctrl.create_from_template("module", "res_probe", "R1.pulses.ro_blank")
    assert str(caught.value) == (
        "Invalid module configuration: Config field 'ch' (Gen ch) expression "
        "'$resonator.wiring.ch' failed: Invalid parameter path "
        "'$resonator.wiring.ch': missing parameter"
    )
    assert _library_keys(ctrl) == before


def test_create_from_template_name_clash_fails(qapp, tmp_path: Path):
    """Create rejects a duplicate without overwriting the first entry."""
    ctrl = _make_ctrl(tmp_path, {"freq": 6000.0})
    ctrl.context_control.set_module(
        ctrl.read_context().target, "R1.pulses.dup", module_spec_from_raw(_pulse_raw())
    )
    with pytest.raises(RuntimeError, match="already exists"):
        ctrl.create_from_template("module", "res_probe", "R1.pulses.dup")
    assert (
        ctrl.read_context().library.get_module("R1.pulses.dup").to_dict()["type"]
        == "pulse"
    )


def test_create_waveform_from_role(qapp, tmp_path: Path):
    ctrl = _make_ctrl(tmp_path)
    ctrl.create_from_template("waveform", "res_waveform", "R1.waveforms.ro_wav")
    assert "R1.waveforms.ro_wav" in ctrl.read_context().library.waveform_keys()


def test_create_from_blank_module_role(qapp, tmp_path: Path):
    """A ':blank' role creates a structural-zero entry of that exact shape."""
    ctrl = _make_ctrl(tmp_path, {"freq": 6000.0})
    ctrl.create_from_template("module", "reset/bath:blank", "R1.reset.rb")
    raw = ctrl.read_context().library.get_module("R1.reset.rb").to_dict()
    assert raw["type"] == "reset/bath"


def test_create_from_blank_waveform_role_uncovered_style(qapp, tmp_path: Path):
    """A waveform style without a calibrated role is reachable via :blank."""
    ctrl = _make_ctrl(tmp_path)
    ctrl.create_from_template("waveform", "drag:blank", "R1.waveforms.dwav")
    raw = ctrl.read_context().library.get_waveform("R1.waveforms.dwav").to_dict()
    assert raw["style"] == "drag"


def test_item_kind_mismatch_raises(qapp, tmp_path: Path):
    ctrl = _make_ctrl(tmp_path)
    with pytest.raises(RuntimeError, match="not a waveform"):
        ctrl.create_from_template("waveform", "res_probe", "x")


def test_unknown_role_raises(qapp, tmp_path: Path):
    ctrl = _make_ctrl(tmp_path)
    before = _library_keys(ctrl)
    with pytest.raises(KeyError):
        ctrl.create_from_template("module", "no_such_role", "x")
    assert _library_keys(ctrl) == before


def test_empty_name_raises(qapp, tmp_path: Path):
    ctrl = _make_ctrl(tmp_path)
    with pytest.raises(RuntimeError, match="name must not be empty"):
        ctrl.create_from_template("module", "res_probe", "")


def test_no_catalog_wired_raises(qapp):
    ctx = SessionEnv(parameters=None, soc=None, soccfg=None)
    ctrl = Controller(
        state=State(ctx),
        registry=Registry(),
        view=None,
        bus=EventBus(),
    )
    with pytest.raises(RuntimeError, match="No template catalog"):
        ctrl.create_from_template("module", "res_probe", "x")
