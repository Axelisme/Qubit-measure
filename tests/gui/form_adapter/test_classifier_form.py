"""Generated readout calibration forms through public FormState operations."""

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import (
    Centroid3Classifier,
    CircleClassifier,
    ConfigBase,
    ConstWaveformCfg,
    LinearClassifier,
    MuxReadoutSpec,
    MuxReadoutTone,
    PulseReadoutCfg,
    PulseReadoutSpec,
    ReadoutClassifier,
    Template,
    TemplateDef,
)
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState
from zcu_tools.gui.form_adapter import Form, module_form


def result_cfg[T: ConfigBase](
    form: Form[T], state: FormState[FormModel], env: FormEnv
) -> T:
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    return cfg


class ReadoutPlan(ConfigBase):
    readout: PulseReadoutCfg


def test_readout_template_can_bind_with_nullable_tagged_children() -> None:
    form = Form(ReadoutPlan, {"readout": Template("synthetic")})
    form.bind_templates(
        {
            "synthetic": TemplateDef(
                PulseReadoutSpec,
                "Synthetic",
                {
                    "gen_ch": 0,
                    "ro_ch": 0,
                    "nqz": 1,
                    "freq": 6000.0,
                    "gain": 0.2,
                    "ro_length": 1.0,
                    "waveform.style": "const",
                    "waveform.length": 1.0,
                },
            )
        },
        adapter="synthetic",
    )
    env = FormEnv()
    state = FormState(form.default_input(env))
    state.select(("readout", "module", "classifier"), {"mode": "linear"}, env)
    state.write(
        {
            ("readout", "module", "classifier", "angle"): 0.5,
            ("readout", "module", "classifier", "threshold"): 2.0,
        },
        env,
    )
    assert result_cfg(form, state, env).readout.classifier == LinearClassifier(
        angle=0.5, threshold=2.0
    )


class CalibrationPlan(ConfigBase):
    calibration: ReadoutClassifier | None = None


def test_native_model_union_selects_linear_then_circle_then_none() -> None:
    env = FormEnv()
    form = Form(CalibrationPlan, {})
    state = FormState(form.default_input(env))
    target = state.describe(("calibration",), env).target
    assert target is not None and target.selected == 0
    state.select(("calibration",), {"mode": "linear"}, env)
    state.write({("calibration", "angle"): 0.5, ("calibration", "threshold"): 2.0}, env)
    assert result_cfg(form, state, env).calibration == LinearClassifier(
        angle=0.5, threshold=2.0
    )
    state.select(("calibration",), {"mode": "circle"}, env)
    assert {
        field.path[-1] for field in state.describe(("calibration",), env).children
    } == {"g_center", "e_center", "radius", "confusion_matrix"}
    state.write(
        {
            ("calibration", "g_center"): {"__complex__": [1.0, 2.0]},
            ("calibration", "e_center"): {"__complex__": [3.0, 4.0]},
        },
        env,
    )
    assert result_cfg(form, state, env).calibration == CircleClassifier(
        g_center=1 + 2j, e_center=3 + 4j
    )
    state.select(("calibration",), {"mode": "linear"}, env)
    state.write(
        {("calibration", "angle"): 0.75, ("calibration", "threshold"): 5.0}, env
    )
    assert result_cfg(form, state, env).calibration == LinearClassifier(
        angle=0.75, threshold=5.0
    )
    state.select(("calibration",), None, env)
    assert state.describe(("calibration",), env).children == ()
    assert state.preview(env).values[("calibration",)] is None


@pytest.mark.parametrize(
    "classifier",
    [
        CircleClassifier(g_center=1 + 2j, e_center=3 + 4j),
        LinearClassifier(angle=0.5, threshold=2.0),
        Centroid3Classifier(
            g_center=1 + 2j,
            e_center=3 + 4j,
            f_center=5 + 6j,
            confusion_matrix=[[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],
        ),
    ],
)
def test_module_draft_preserves_existing_classifier_and_switches_branches(
    classifier: ReadoutClassifier,
) -> None:
    env = FormEnv()
    pulse = PulseReadoutSpec(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=6000.0,
        gain=0.2,
        ro_length=1.0,
        waveform=ConstWaveformCfg(length=1.0),
        classifier=classifier,
    )
    pulse_form = module_form(PulseReadoutSpec)
    pulse_state = FormState(pulse_form.input_from_spec(pulse, env))
    assert pulse_state.describe(("classifier",), env).children
    spec = MuxReadoutSpec(
        gen_ch=0,
        nqz=1,
        mixer_freq=6000.0,
        length=1.0,
        ro_length=1.0,
        tones=[
            MuxReadoutTone(freq=6000.0, gain=0.2, ro_ch=0, classifier=classifier),
            MuxReadoutTone(freq=6100.0, gain=0.3, ro_ch=1),
        ],
    )
    form = module_form(MuxReadoutSpec)
    state = FormState(form.input_from_spec(spec, env))
    assert state.describe(("tones", 0, "classifier"), env).children
    state.select(("tones", 1, "classifier"), {"mode": "linear"}, env)
    state.write(
        {
            ("tones", 1, "classifier", "angle"): 0.75,
            ("tones", 1, "classifier", "threshold"): 4.0,
        },
        env,
    )
    cfg = result_cfg(form, state, env)
    assert cfg.tones[0].classifier == classifier
    assert cfg.tones[1].classifier == LinearClassifier(angle=0.75, threshold=4.0)
    state.select(
        ("tones", 0, "classifier"),
        {"mode": "circle" if classifier.mode == "linear" else "linear"},
        env,
    )
    assert {
        field.path[-1]
        for field in state.describe(("tones", 0, "classifier"), env).children
    } == (
        {"g_center", "e_center", "radius", "confusion_matrix"}
        if classifier.mode == "linear"
        else {"angle", "threshold", "confusion_matrix"}
    )


def test_classifier_switch_inherits_raw_matrix_and_reports_target_shape_errors() -> (
    None
):
    env = FormEnv()
    form = Form(CalibrationPlan, {})
    state = FormState(form.default_input(env))
    state.select(("calibration",), {"mode": "circle"}, env)
    matrix: JsonValue = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    state.write(
        {
            ("calibration", "confusion_matrix"): matrix,
            ("calibration", "g_center"): {"expr": "1 + 2j"},
        },
        env,
    )
    authored = state.input
    state.select(("calibration",), {"mode": "circle"}, env)
    assert state.input.same_as(authored)
    state.select(("calibration",), {"mode": "linear"}, env)
    assert state.input.data["calibration"] == {
        "mode": "linear",
        "confusion_matrix": matrix,
    }
    state.write(
        {("calibration", "angle"): 0.25, ("calibration", "threshold"): 3.0}, env
    )
    errors = state.preview(env).errors
    assert errors is not None
    assert any(
        path[:2] == ("calibration", "confusion_matrix") for path in errors.messages
    )
    state.write({("calibration", "confusion_matrix"): [[1.0, 0.0], [0.0, 1.0]]}, env)
    assert not isinstance(state.result(env), FormErrors)
    state.select(("calibration",), {"mode": "circle"}, env)
    assert {
        field.path[-1] for field in state.describe(("calibration",), env).children
    } == {"g_center", "e_center", "radius", "confusion_matrix"}
    assert state.input.data["calibration"] == {
        "mode": "circle",
        "confusion_matrix": [[1.0, 0.0], [0.0, 1.0]],
    }


def test_default_none_is_selected_and_circle_can_be_edited() -> None:
    env = FormEnv()
    form = module_form(PulseReadoutSpec)
    state = FormState(form.default_input(env))
    target = state.describe(("classifier",), env).target
    assert target is not None and target.options is not None
    assert target.selected == 0
    assert target.options[0].label == "None"
    state.select(("classifier",), {"mode": "circle"}, env)
    assert {
        field.path[-1] for field in state.describe(("classifier",), env).children
    } == {"g_center", "e_center", "radius", "confusion_matrix"}
    state.write(
        {
            ("classifier", "g_center"): {"__complex__": [1.0, 2.0]},
            ("classifier", "e_center"): {"__complex__": [3.0, 4.0]},
        },
        env,
    )
    # Complete the unrelated required module fields before lowering the draft.
    state.write(
        {
            ("gen_ch",): 0,
            ("ro_ch",): 0,
            ("nqz",): 1,
            ("freq",): 6000.0,
            ("gain",): 0.2,
            ("ro_length",): 1.0,
            ("waveform",): {
                "kind": "inline",
                "module": {"style": "const", "length": 1.0},
            },
        },
        env,
    )
    assert result_cfg(form, state, env).classifier == CircleClassifier(
        g_center=1 + 2j, e_center=3 + 4j
    )
