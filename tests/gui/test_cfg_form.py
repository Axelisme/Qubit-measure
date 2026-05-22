"""Tests — CfgFormWidget populate / read_values round-trip (Phase 19)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    schema_to_dict,
)
from zcu_tools.gui.event_bus import EventBus, GuiEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bus():
    return EventBus()


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _schema(spec_fields: dict, value_fields: dict) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),
        value=CfgSectionValue(fields=value_fields),
    )


def _make_ctx():
    from zcu_tools.gui.adapter import ExpContext

    return ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)


# ---------------------------------------------------------------------------
# schema_to_dict — SweepValue modes
# ---------------------------------------------------------------------------


def test_sweep_value_expts_mode():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=1.0, stop=2.0, expts=5)},
    )
    result = schema_to_dict(schema, ml)
    assert isinstance(result["f"], SweepCfg)
    assert result["f"].expts == 5


def test_sweep_value_step_mode():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    result = schema_to_dict(schema, ml)
    sweep = result["f"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.step == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# make_scalar_widget / read_scalar_widget
# ---------------------------------------------------------------------------


def test_scalar_int_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="X", type=int)
    w = make_scalar_widget(spec, 42)
    assert read_scalar_widget(w, spec) == 42


def test_scalar_float_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Pi", type=float)
    w = make_scalar_widget(spec, 3.14)
    assert read_scalar_widget(w, spec) == pytest.approx(3.14)


def test_scalar_bool_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Flag", type=bool)
    w = make_scalar_widget(spec, True)
    assert read_scalar_widget(w, spec) is True


def test_scalar_choices_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Model", type=str, choices=["hm", "t", "auto"])
    w = make_scalar_widget(spec, "hm")
    assert read_scalar_widget(w, spec) == "hm"


def test_scalar_editable_false_widget_disabled(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget

    spec = ScalarSpec(label="RO", type=float, editable=False)
    w = make_scalar_widget(spec, 1.0)
    assert not w.isEnabled()


# ---------------------------------------------------------------------------
# CfgFormWidget — populate and read_values / read_schema
# ---------------------------------------------------------------------------


def test_read_values_before_populate_raises(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_values()


def test_read_schema_before_populate_raises(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_schema()


def test_populate_scalar_fields_round_trip(qapp, bus):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "reps": ScalarSpec(label="Reps", type=int),
            "freq": ScalarSpec(label="Freq", type=float),
        },
        {
            "reps": ScalarValue(100),
            "freq": ScalarValue(6.0),
        },
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    assert out.fields["reps"].value == 100  # type: ignore[union-attr]
    assert out.fields["freq"].value == pytest.approx(6.0)  # type: ignore[union-attr]


def test_read_schema_returns_cfg_schema(qapp, bus):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": ScalarValue(10)},
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_schema()
    assert isinstance(out, CfgSchema)
    assert out.spec is schema.spec


def test_read_values_does_not_mutate_original(qapp, bus):
    from qtpy.QtWidgets import QSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": ScalarValue(100)},
    )
    w = CfgFormWidget()
    w.populate(schema, bus)

    spin = w.findChild(QSpinBox)
    assert spin is not None
    spin.setValue(999)

    out = w.read_values()
    assert out.fields["reps"].value == 999  # type: ignore[union-attr]
    assert schema.value.fields["reps"].value == 100  # type: ignore[union-attr]


def test_populate_sweep_field_round_trip(qapp, bus):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=5.8, stop=6.2, expts=201)},
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.start == pytest.approx(5.8)
    assert sv.stop == pytest.approx(6.2)
    assert sv.expts == 201
    assert sv.step is None


def test_populate_sweep_field_step_preserved(qapp, bus):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.1)


def test_populate_nested_section_round_trip(qapp, bus):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "inner": CfgSectionSpec(
                fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        },
        {"inner": CfgSectionValue(fields={"gain": ScalarValue(0.05)})},
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    inner = out.fields["inner"]
    assert isinstance(inner, CfgSectionValue)
    assert inner.fields["gain"].value == pytest.approx(0.05)  # type: ignore[union-attr]


def test_populate_multi_sweep_round_trip(qapp, bus):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "ms": MultiSweepSpec(
                axes={"x": SweepSpec(label="X"), "y": SweepSpec(label="Y")},
                label="Multi",
            )
        },
        {
            "ms": MultiSweepValue(
                axes={
                    "x": SweepValue(start=0.0, stop=1.0, expts=5),
                    "y": SweepValue(start=2.0, stop=3.0, expts=3),
                }
            )
        },
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    ms = out.fields["ms"]
    assert isinstance(ms, MultiSweepValue)
    assert ms.axes["x"].expts == 5
    assert ms.axes["y"].start == pytest.approx(2.0)


def test_populate_module_ref_field_round_trip(qapp, bus):
    from zcu_tools.gui.adapter import ModuleRefSpec, ModuleRefValue
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    allowed_spec = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"mod": ModuleRefSpec(allowed=[allowed_spec], label="Module")}
        ),
        value=CfgSectionValue(
            fields={
                "mod": ModuleRefValue(
                    chosen_key="<Custom:Pulse>",
                    value=CfgSectionValue(fields={"gain": ScalarValue(0.5)}),
                )
            }
        ),
    )
    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    mod = out.fields["mod"]
    assert isinstance(mod, ModuleRefValue)
    assert mod.chosen_key == "<Custom:Pulse>"
    assert mod.value.fields["gain"].value == pytest.approx(0.5)  # type: ignore[union-attr]


def test_populate_full_fake_freq_schema(qapp, bus):
    """Smoke test: FakeFreqAdapter default schema populates and round-trips."""
    from zcu_tools.experiment.v2_gui.adapters.onetone.freq import FakeFreqAdapter
    from zcu_tools.gui.adapter import ModuleRefSpec
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)

    w = CfgFormWidget()
    w.populate(schema, bus)
    out = w.read_values()

    for key in ("reps", "rounds", "freq", "res_freq", "Ql", "modules"):
        assert key in out.fields, f"missing key: {key}"

    assert isinstance(out.fields["freq"], SweepValue)
    # modules is a CfgSectionValue with readout as ModuleRefValue
    modules_val = out.fields["modules"]
    assert isinstance(modules_val, CfgSectionValue)
    # Verify spec has ModuleRefSpec for readout
    modules_spec = schema.spec.fields["modules"]
    assert hasattr(modules_spec, "fields")
    readout_spec = modules_spec.fields["readout"]  # type: ignore[union-attr]
    assert isinstance(readout_spec, ModuleRefSpec)
