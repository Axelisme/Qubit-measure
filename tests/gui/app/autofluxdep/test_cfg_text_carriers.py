"""Direct text carriers in autofluxdep sweep knobs project to numeric values.

Sweep and centered-sweep controls may hold direct text input. The node schema
projects valid text to numeric knobs, keeps locked centers intact, and blocks
lowering when a carrier holds invalid text.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.experiments.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.cfg import CenteredSweepValue, DirectValue, SweepValue

from ._helpers import sectioned_test_schema


def test_sweep_direct_text_carriers_project_to_numeric_knobs():
    schema = sectioned_test_schema()
    schema.set_field(
        "detune_sweep",
        SweepValue(
            DirectValue(-2.0, raw="-2.00"),
            DirectValue(2.0, raw="2e0"),
            5,
        ),
    )

    assert schema.read_knobs()["detune_sweep"] == {
        "start": -2.0,
        "stop": 2.0,
        "expts": 5,
    }


def test_locked_center_accepts_direct_text_carrier_without_losing_numeric_projection():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field(
        "detune_sweep",
        CenteredSweepValue(DirectValue(0.0, raw="0.00"), span=20.0, expts=5),
    )

    assert schema.read_knobs()["detune_sweep"] == {
        "center": 0.0,
        "span": 20.0,
        "expts": 5,
        "step": 5.0,
    }
    detune = schema.lower(None)["detune_sweep"]
    assert float(detune.start) == pytest.approx(-10.0)
    assert float(detune.stop) == pytest.approx(10.0)


def test_centered_sweep_control_carriers_project_numeric_knobs_and_block_invalid_lowering():
    schema = QubitFreqBuilder().make_default_schema()
    schema.set_field(
        "detune_sweep",
        CenteredSweepValue(
            0.0,
            span=DirectValue(20.0, raw="20.00"),
            expts=DirectValue(5, raw="005"),
            step=DirectValue(5.0, raw="5e0"),
        ),
    )
    assert schema.read_knobs()["detune_sweep"] == {
        "center": 0.0,
        "span": 20.0,
        "expts": 5,
        "step": 5.0,
    }
    detune = schema.lower(None)["detune_sweep"]
    assert detune.expts == 5
    assert float(detune.start) == pytest.approx(-10.0)
    assert float(detune.stop) == pytest.approx(10.0)

    invalid = CenteredSweepValue(
        0.0,
        span=DirectValue(None, raw="1e", error="Invalid number"),
        expts=5,
    )
    schema.set_field("detune_sweep", invalid)
    assert schema.read_knobs()["detune_sweep"]["span"] is None
    with pytest.raises(RuntimeError, match="span"):
        schema.lower(None)


def test_locked_center_rejects_invalid_direct_text_carrier():
    schema = QubitFreqBuilder().make_default_schema()
    before = schema.read_knobs()
    with pytest.raises(ValueError, match="invalid center"):
        schema.set_field(
            "detune_sweep",
            CenteredSweepValue(
                DirectValue(None, raw="-", error="invalid center"),
                span=20.0,
                expts=5,
            ),
        )
    assert schema.read_knobs() == before
