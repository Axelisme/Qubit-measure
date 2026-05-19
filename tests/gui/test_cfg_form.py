"""Phase 10 tests — CfgFormWidget build and read_schema round-trip."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSection,
    ModuleRefField,
    MultiSweepField,
    ScalarField,
    SweepField,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def _schema(fields: dict) -> CfgSchema:
    return CfgSchema(root=CfgSection(fields=fields))


def _make_ctx():
    from zcu_tools.gui.adapter import ExpContext

    return ExpContext(
        md=MagicMock(), ml=MagicMock(), em=MagicMock(), soc=None, soccfg=None
    )


# ---------------------------------------------------------------------------
# SweepField schema_to_dict — step mode (10a)
# ---------------------------------------------------------------------------


def test_sweep_field_step_mode_schema_to_dict():
    from zcu_tools.gui.adapter import schema_to_dict
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema({"f": SweepField(start=0.0, stop=1.0, expts=11, step=0.1)})
    result = schema_to_dict(schema, ml)
    sweep = result["f"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.start == pytest.approx(0.0)
    assert sweep.stop == pytest.approx(1.0)
    assert sweep.step == pytest.approx(0.1)


def test_sweep_field_expts_mode_unchanged():
    from zcu_tools.gui.adapter import schema_to_dict
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema({"f": SweepField(start=1.0, stop=2.0, expts=5)})
    result = schema_to_dict(schema, ml)
    assert isinstance(result["f"], SweepCfg)
    assert result["f"].expts == 5


# ---------------------------------------------------------------------------
# make_scalar_widget / read_scalar_widget
# ---------------------------------------------------------------------------


def test_scalar_int_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    field = ScalarField(value=42, label="X", type=int)
    w = make_scalar_widget(field)
    assert read_scalar_widget(w, field) == 42


def test_scalar_float_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    field = ScalarField(value=3.14, label="Pi", type=float)
    w = make_scalar_widget(field)
    assert read_scalar_widget(w, field) == pytest.approx(3.14)


def test_scalar_bool_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    field = ScalarField(value=True, label="Flag", type=bool)
    w = make_scalar_widget(field)
    assert read_scalar_widget(w, field) is True


def test_scalar_choices_widget_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget, read_scalar_widget

    field = ScalarField(
        value="hm", label="Model", type=str, choices=["hm", "t", "auto"]
    )
    w = make_scalar_widget(field)
    assert read_scalar_widget(w, field) == "hm"


def test_scalar_editable_false_widget_disabled(qapp):
    from zcu_tools.gui.ui.cfg_form import make_scalar_widget

    field = ScalarField(value=1.0, label="RO", type=float, editable=False)
    w = make_scalar_widget(field)
    assert not w.isEnabled()


# ---------------------------------------------------------------------------
# CfgFormWidget — populate and read_schema
# ---------------------------------------------------------------------------


def test_read_schema_before_populate_raises(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_schema()


def test_populate_scalar_fields_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "reps": ScalarField(value=100, label="Reps", type=int),
            "freq": ScalarField(value=6.0, label="Freq", type=float),
        }
    )
    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    assert out.root.fields["reps"].value == 100  # type: ignore[union-attr]
    assert out.root.fields["freq"].value == pytest.approx(6.0)  # type: ignore[union-attr]


def test_read_schema_does_not_mutate_original(qapp):
    from qtpy.QtWidgets import QSpinBox  # type: ignore[attr-defined]

    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema({"reps": ScalarField(value=100, label="Reps", type=int)})
    w = CfgFormWidget()
    w.populate(schema)

    spin = w.findChild(QSpinBox)
    assert spin is not None
    spin.setValue(999)

    out = w.read_schema()
    assert out.root.fields["reps"].value == 999  # type: ignore[union-attr]
    assert schema.root.fields["reps"].value == 100  # original unchanged


def test_populate_sweep_field_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema({"f": SweepField(start=5.8, stop=6.2, expts=201)})
    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    sf = out.root.fields["f"]
    assert isinstance(sf, SweepField)
    assert sf.start == pytest.approx(5.8)
    assert sf.stop == pytest.approx(6.2)
    assert sf.expts == 201
    assert sf.step is None


def test_populate_sweep_field_step_preserved(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema({"f": SweepField(start=0.0, stop=1.0, expts=11, step=0.1)})
    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    sf = out.root.fields["f"]
    assert isinstance(sf, SweepField)
    assert sf.step == pytest.approx(0.1)


def test_populate_nested_section_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "inner": CfgSection(
                fields={
                    "gain": ScalarField(value=0.05, label="Gain", type=float),
                }
            )
        }
    )
    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    inner = out.root.fields["inner"]
    assert isinstance(inner, CfgSection)
    assert inner.fields["gain"].value == pytest.approx(0.05)  # type: ignore[union-attr]


def test_populate_multi_sweep_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "ms": MultiSweepField(
                sweeps={
                    "x": SweepField(start=0.0, stop=1.0, expts=5),
                    "y": SweepField(start=2.0, stop=3.0, expts=3),
                }
            )
        }
    )
    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    ms = out.root.fields["ms"]
    assert isinstance(ms, MultiSweepField)
    assert ms.sweeps["x"].expts == 5
    assert ms.sweeps["y"].start == pytest.approx(2.0)


def test_populate_module_ref_field_round_trip(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "mod": ModuleRefField(
                module_name="pulse_a",
                override={},
                inline_cfg=None,
                expanded_content=None,
                available_modules=["pulse_a", "pulse_b"],
            )
        }
    )
    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    mod = out.root.fields["mod"]
    assert isinstance(mod, ModuleRefField)
    assert mod.module_name == "pulse_a"


def test_populate_full_fake_freq_schema(qapp):
    """Smoke test: FakeFreqAdapter default schema populates and round-trips."""
    from zcu_tools.experiment.v2_gui.adapters.onetone.freq import FakeFreqAdapter
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)

    w = CfgFormWidget()
    w.populate(schema)
    out = w.read_schema()

    for key in (
        "reps",
        "rounds",
        "freq_start",
        "freq_stop",
        "freq_expts",
        "freq",
        "Ql",
    ):
        assert key in out.root.fields
