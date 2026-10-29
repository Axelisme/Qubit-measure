"""Readout calibration schemas use normalized, offset-free ADC coordinates."""

import pytest
from pydantic import ValidationError
from zcu_tools import cfg_model
from zcu_tools.cfg_model import CircleClassifier, MuxReadoutTone, PulseReadoutSpec


def test_pulse_readout_accepts_circle_centers_and_encodes_complex_tags():
    spec = PulseReadoutSpec.model_validate(
        {
            "gen_ch": 0,
            "nqz": 1,
            "freq": 700.0,
            "gain": 0.2,
            "waveform": {"style": "const", "length": 1.0},
            "ro_ch": 1,
            "ro_length": 1.5,
            "classifier": {
                "mode": "circle",
                "g_center": 1.0 - 2.0j,
                "e_center": -3.0 + 4.0j,
            },
        }
    )

    assert spec.to_dict()["classifier"] == {
        "mode": "circle",
        "g_center": {"__complex__": [1.0, -2.0]},
        "e_center": {"__complex__": [-3.0, 4.0]},
    }
    assert PulseReadoutSpec.model_validate(spec.to_dict()) == spec


def test_centroid3_preserves_three_centers_and_training_matrix():
    matrix = [[1.2, -0.2, 0.1], [0.3, 0.6, 0.2], [0.0, 0.1, 0.8]]
    raw = {
        "mode": "centroid3",
        "g_center": {"__complex__": [1.0, -2.0]},
        "e_center": {"__complex__": [-3.0, 4.0]},
        "f_center": {"__complex__": [5.0, 6.0]},
        "confusion_matrix": matrix,
    }
    tone = MuxReadoutTone.model_validate(
        {"freq": 6020.0, "gain": 0.1, "ro_ch": 3, "classifier": raw}
    )
    assert tone.to_dict()["classifier"] == raw
    assert isinstance(tone.classifier, cfg_model.Centroid3Classifier)
    assert tone.classifier.f_center == 5.0 + 6.0j
    assert MuxReadoutTone.model_validate(tone.to_dict()) == tone


@pytest.mark.parametrize(
    "override",
    [
        {"g_center": complex(float("nan"), 0)},
        {"e_center": complex(0, float("inf"))},
        {"f_center": {"__complex__": [float("inf"), 0.0]}},
        {"g_center": 2j},
        {"f_center": 2j},
        {"f_center": 1j},
        {"confusion_matrix": None},
        {"confusion_matrix": [[1.0, 0.0]] * 3},
        {"confusion_matrix": [[1.0, 0.0, 0.0]] * 2},
        {"confusion_matrix": [[1.0, 0.0, float("nan")]] * 3},
    ],
)
def test_centroid3_rejects_invalid_centers_and_matrix(override: dict[str, object]):
    with pytest.raises(ValidationError):
        cfg_model.Centroid3Classifier.model_validate(
            {
                "g_center": 1j,
                "e_center": 2j,
                "f_center": 3j,
                "confusion_matrix": [[1.0, 0.0, 0.0]] * 3,
                **override,
            }
        )


def test_centroid3_requires_training_matrix():
    with pytest.raises(ValidationError, match="confusion_matrix"):
        cfg_model.Centroid3Classifier.model_validate(
            {"g_center": 1j, "e_center": 2j, "f_center": 3j}
        )


def test_centroid3_typed_spec_can_be_materialized_without_losing_tags():
    calibration = cfg_model.Centroid3Classifier(
        g_center=1j,
        e_center=2j,
        f_center=3j,
        confusion_matrix=[[1.2, -0.2, 0.0], [0.1, 0.8, 0.2], [0.0, 0.0, 0.9]],
    )
    spec = PulseReadoutSpec(
        gen_ch=0,
        nqz=1,
        freq=100.0,
        gain=0.1,
        ro_ch=1,
        ro_length=1.5,
        waveform=cfg_model.ConstWaveformCfg(length=1.0),
        classifier=calibration,
    )
    validated = cfg_model.module_spec_from_raw(spec)
    materialized = cfg_model.ModuleCfgFactory.from_raw(validated)
    assert materialized.to_dict()["classifier"] == calibration.to_dict()


def test_mux_tone_accepts_linear_calibration_without_rotating_or_normalizing():
    matrix = [[1.2, -0.2], [0.1, 0.7]]
    tone = MuxReadoutTone.model_validate(
        {
            "freq": 6020.0,
            "gain": 0.1,
            "ro_ch": 3,
            "classifier": {
                "mode": "linear",
                "angle": -0.75,
                "threshold": -1.25,
                "confusion_matrix": matrix,
            },
        }
    )
    assert tone.to_dict()["classifier"] == {
        "mode": "linear",
        "angle": -0.75,
        "threshold": -1.25,
        "confusion_matrix": matrix,
    }
    assert cfg_model.LinearClassifier(angle=0.5, threshold=0.0).confusion_matrix is None
    assert MuxReadoutTone.model_validate(tone.to_dict()) == tone


@pytest.mark.parametrize(
    "override",
    [
        {"angle": float("nan")},
        {"threshold": float("inf")},
        {"confusion_matrix": [[1.0, 0.0]]},
        {"confusion_matrix": [[1.0, 0.0], [0.0, float("inf")]]},
        {"confusion_matrix": [[1.0, 0.0, 0.0]] * 3},
        {"mode": "unknown"},
    ],
)
def test_linear_rejects_nonfinite_values_or_wrong_shape(override: dict[str, object]):
    with pytest.raises(ValidationError):
        cfg_model.LinearClassifier.model_validate(
            {"angle": 0.5, "threshold": 0.1, **override}
        )


def test_circle_optional_post_calibration_preserves_matrix_without_normalization():
    primary = CircleClassifier(g_center=1.0j, e_center=2.0j)
    assert primary.radius is None
    assert primary.confusion_matrix is None
    # The schema stores calibration estimates, not just stochastic matrices.
    matrix = [[1.1, -0.1, 0.2], [0.3, 0.9, 0.0], [0.0, 0.0, 0.8]]
    complete = primary.with_updates(radius=0.25, confusion_matrix=matrix)
    assert complete.radius == 0.25
    assert complete.confusion_matrix == matrix
    assert CircleClassifier.model_validate(complete.to_dict()) == complete


@pytest.mark.parametrize(
    "override",
    [
        {"radius": 0.0},
        {"radius": -1.0},
        {"radius": float("inf")},
        {"radius": float("nan")},
        {"g_center": complex(float("inf"), 0.0)},
        {"e_center": {"__complex__": [0.0, float("nan")]}},
        {"confusion_matrix": [[1.0, 0.0], [0.0, 1.0]]},
        {"confusion_matrix": [[1.0, 0.0, 0.0]] * 2 + [[0.0, 1.0]]},
        {"confusion_matrix": [[1.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, float("nan")]]},
        {"g_center": {"__complex__": [True, 0.0]}},
        {"g_center": {"__complex__": [1.0, 2.0], "extra": 0}},
    ],
)
def test_circle_rejects_invalid_geometry_or_matrix(override: dict[str, object]):
    with pytest.raises(ValidationError):
        CircleClassifier.model_validate(
            {"g_center": 1.0j, "e_center": 2.0j, **override}
        )
