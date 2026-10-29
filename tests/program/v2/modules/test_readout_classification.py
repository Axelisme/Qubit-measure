"""Readout calibration becomes explicit, per-ADC acquisition parameters."""

import pytest
from zcu_tools.cfg_model import CircleClassifier, LinearClassifier
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


def test_tone_parameters_keep_adc_numbers_not_tone_positions():
    from tests.program.v2._mux_support import native_mux

    cfg = native_mux((4, 1))
    first = CircleClassifier(g_center=0j, e_center=1 + 0j, radius=0.5)
    second = LinearClassifier(angle=1.0, threshold=2.0)
    cfg.tones[0].classifier = first
    cfg.tones[1].classifier = second
    assert readout.classification_parameters(cfg) == {4: first, 1: second}


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


def test_readout_supplies_explicit_adc_classification():
    calibration = CircleClassifier(g_center=0j, e_center=1 + 0j, radius=0.5)
    assert readout.classification_parameters(pulse(calibration)) == {2: calibration}


@pytest.mark.parametrize(
    "classifier, message",
    [
        (None, "classifier.*missing"),
        (CircleClassifier(g_center=0j, e_center=1 + 0j), "radius.*Post"),
        (LinearClassifier(angle=0.0, threshold=0.5), "linear.*other"),
    ],
)
def test_other_consumer_requires_complete_circle(classifier, message):
    with pytest.raises(ValueError, match=message):
        readout.classification_parameters(pulse(classifier), require_other=True)
