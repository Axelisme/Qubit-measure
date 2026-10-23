"""Generated Form errors retain authored field guidance at canonical paths."""

from typing import Annotated

import pytest
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2 import SweepCfg


class GuidedInputs(ConfigBase):
    """A synthetic adapter needs one calibration and an explicit scan window."""

    gain: Annotated[
        float,
        FieldMeta(
            label="F1 drive gain",
            unit="normalized",
            description="Enter the signed amplitude from the Rabi calibration.",
        ),
    ]
    frequency: Annotated[
        SweepCfg,
        FieldMeta(
            label="Probe frequency",
            unit="MHz",
            description="Enter absolute endpoints before the survey.",
        ),
    ]
    plain: float


@pytest.mark.parametrize("blank", ["absent", "null"])
def test_required_errors_include_authored_guidance_without_changing_paths(
    blank: str,
) -> None:
    env = FormEnv()
    form = Form(GuidedInputs, {"frequency": Range(None, None, expts=3)})
    state = FormState(form.default_input(env))
    if blank == "absent":
        state.write({("frequency",): {"kind": "linrange", "n": 3}}, env)
    original = state.input
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert set(result.messages) == {
        ("gain",),
        ("frequency", "start"),
        ("frequency", "stop"),
        ("plain",),
    }
    assert "F1 drive gain" in result.messages[("gain",)]
    assert "normalized" in result.messages[("gain",)]
    assert "Rabi calibration" in result.messages[("gain",)]
    for endpoint in ("start", "stop"):
        message = result.messages[("frequency", endpoint)]
        assert "Probe frequency" in message
        assert "MHz" in message
        assert endpoint.capitalize() in message
        assert "absolute endpoints" in message
    assert result.messages[("plain",)] == "Field required"
    preview = state.preview(env)
    assert preview.errors == result
    assert state.input.same_as(original)
    state.write(
        {
            ("gain",): 0.2,
            ("frequency", "start"): 1.0,
            ("frequency", "stop"): 2.0,
            ("plain",): 4.0,
        },
        env,
    )
    accepted = state.result(env)
    assert not isinstance(accepted, FormErrors)
    cfg = form.to_cfg(accepted, env)
    assert isinstance(cfg, GuidedInputs)
    assert cfg.gain == 0.2


@pytest.mark.parametrize(
    "kind,coordinates",
    [
        ("linrange", ("start", "stop")),
        ("center", ("center", "span")),
        ("linrange_step", ("start", "stop", "step")),
        ("center_step", ("center", "span", "step")),
    ],
)
def test_sweep_coordinates_inherit_units_but_counts_do_not(
    kind: str,
    coordinates: tuple[str, ...],
) -> None:
    env = FormEnv()
    form = Form(GuidedInputs, {})
    state = FormState(FormInput(form.model, {"frequency": {"kind": kind}}))
    fields = {
        field.name: field for field in state.describe(("frequency",), env).children
    }
    for coordinate in coordinates:
        assert fields[coordinate].unit == "MHz"
    if "n" in fields:
        assert fields["n"].title == "Expts"
        assert fields["n"].unit is None
