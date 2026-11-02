"""Readout calibration lookup contracts."""

import pytest
from zcu_tools.cfg_model import Centroid3Classifier, CircleClassifier, LinearClassifier
from zcu_tools.program.v2.modules import readout
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


def pulse(classifier):
    return readout.PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        freq=100.0,
        ro_ch=2,
        ro_length=1.0,
        waveform=ConstWaveformCfg(length=1.0),
        classifier=classifier,
    )


def test_circle_lookup_uses_adc_and_accepts_centers_without_radius():
    from tests.program.v2._mux_support import native_mux

    first = CircleClassifier(g_center=1 + 2j, e_center=3 + 4j)
    second = CircleClassifier(g_center=5 + 6j, e_center=7 + 8j)
    cfg = native_mux((4, 1))
    cfg.tones[0].classifier = first
    cfg.tones[1].classifier = second
    assert readout.circle_classifier(cfg, ro_ch=1) == second
    assert readout.circle_classifier(cfg, ro_ch=4) == first
    assert readout.circle_classifier(pulse(first), ro_ch=2) == first
    detached = readout.circle_classifier(cfg, ro_ch=4)
    detached.g_center = 99j
    assert cfg.tones[0].classifier.g_center == 1 + 2j


@pytest.mark.parametrize(
    "classifier, ro_ch, message",
    [
        (None, 2, "ADC 2: classifier missing.*GE"),
        (LinearClassifier(angle=0.0, threshold=0.5), 2, "ADC 2: linear.*centers"),
        (CircleClassifier(g_center=0j, e_center=1j), 1, "ADC 1: not declared"),
    ],
)
def test_circle_lookup_rejects_unavailable_centers(classifier, ro_ch, message):
    with pytest.raises(ValueError, match=message):
        readout.circle_classifier(pulse(classifier), ro_ch=ro_ch)


def centroid3():
    return Centroid3Classifier(
        g_center=0j,
        e_center=2 + 0j,
        f_center=2j,
        confusion_matrix=[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]],
    )


def test_centroid3_lookup_is_detached_and_uses_adc_numbers():
    from tests.program.v2._mux_support import native_mux

    cfg = native_mux((4, 1))
    first = centroid3()
    second = first.with_updates(f_center=3j)
    cfg.tones[0].classifier = first
    cfg.tones[1].classifier = second
    assert readout.centroid3_classifier(cfg, ro_ch=1) == second
    assert readout.centroid3_classifier(pulse(first), ro_ch=2) == first
    detached = readout.centroid3_classifier(cfg, ro_ch=4)
    detached.confusion_matrix[0][0] = -99.0
    detached.f_center = 99j
    assert readout.centroid3_classifier(cfg, ro_ch=4) == first
    assert readout.centroid3_classifier(cfg, ro_ch=1) == second


@pytest.mark.parametrize(
    "classifier, ro_ch, message",
    [
        (None, 2, "ADC 2: classifier missing"),
        (CircleClassifier(g_center=0j, e_center=1j), 2, "ADC 2: circle.*centroid3"),
        (LinearClassifier(angle=0.0, threshold=0.0), 2, "ADC 2: linear.*centroid3"),
        (None, 7, "ADC 7: not declared"),
    ],
)
def test_centroid3_lookup_rejects_missing_wrong_type_and_unknown_adc(
    classifier, ro_ch, message
):
    with pytest.raises(ValueError, match=message):
        readout.centroid3_classifier(pulse(classifier), ro_ch=ro_ch)


@pytest.mark.parametrize("mux", [False, True])
def test_circle_requirement_accepts_complete_calibration_for_every_adc(mux):
    from tests.program.v2._mux_support import native_mux

    circle = CircleClassifier(g_center=0j, e_center=2 + 0j, radius=1.0)
    cfg = native_mux((4, 1)) if mux else pulse(circle)
    if isinstance(cfg, readout.MuxReadoutCfg):
        for tone in cfg.tones:
            tone.classifier = circle.model_copy(deep=True)
    original = cfg.model_copy(deep=True)
    readout.require_circle_classifiers(cfg)
    assert cfg == original


@pytest.mark.parametrize("bad_adc", [2, 4, 1])
@pytest.mark.parametrize(
    ("classifier", "message"),
    [
        (None, "classifier missing"),
        (LinearClassifier(angle=0.0, threshold=0.5), "linear.*not circle"),
        (centroid3(), "centroid3.*not circle"),
        (CircleClassifier(g_center=0j, e_center=2 + 0j), "radius missing.*GE Post"),
    ],
)
def test_circle_requirement_rejects_incomplete_or_wrong_mode_on_any_adc(
    bad_adc, classifier, message
):
    from tests.program.v2._mux_support import native_mux

    cfg = pulse(classifier) if bad_adc == 2 else native_mux((4, 1))
    if isinstance(cfg, readout.MuxReadoutCfg):
        for tone in cfg.tones:
            tone.classifier = (
                classifier
                if tone.ro_ch == bad_adc
                else CircleClassifier(g_center=0j, e_center=2 + 0j, radius=1.0)
            )
    with pytest.raises(ValueError, match=rf"ADC {bad_adc}: {message}"):
        readout.require_circle_classifiers(cfg)


def test_centroid3_is_not_circle_calibration():
    with pytest.raises(ValueError, match="centroid3.*circle"):
        readout.circle_classifier(pulse(centroid3()), ro_ch=2)
