"""Readout calibration lookup and native QICK projection contracts."""

import numpy as np
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


@pytest.mark.parametrize("mode", ["linear", "circle"])
def test_qick_options_project_native_shots_without_offset_and_match_shared_domain(
    monkeypatch: pytest.MonkeyPatch, mode: str
):
    from tests.program._acquisition_soc import make_chunk_acquisition

    calibration = (
        LinearClassifier(angle=-np.pi / 2, threshold=1.0)
        if mode == "linear"
        else CircleClassifier(g_center=2 - 1j, e_center=2 + 1j, radius=0.25)
    )
    cfg = pulse(calibration)
    cfg.ro_ch = 0
    iq = np.array([[[2, -1]], [[2, 2]], [[2, 1]], [[2, 8]]], dtype=np.float64)
    if mode == "circle":
        iq[1, 0, 1] = 1
        iq[2, 0, 1] = 0
    else:
        # Zero I makes threshold equality exact despite cos(pi/2) roundoff.
        iq[:, 0, 0] = 0
    soc, program = make_chunk_acquisition(reps=1, reads=1, sweep=[("point", 4)])
    program.soccfg["readouts"][0]["iq_offset"] = 17.0
    raw = (iq * program.ro_chs[0]["length"]).astype(np.int64)
    monkeypatch.setattr(soc, "poll_data", lambda: [(4, ([raw.reshape(-1, 2)], {}))])
    options = readout.qick_threshold_options(cfg)
    projected = program.acquire(soc, progress=False, **options)
    assert projected is not None
    expected = [0, 1, 0, 1]
    np.testing.assert_array_equal(projected[0][0, :, 0], expected)
    np.testing.assert_array_equal(projected[0][0, :, 1], 0)
    classified = readout.classification_processor(calibration)(iq)[:, 0]
    shared = 4 if mode == "linear" else 2
    np.testing.assert_array_equal(projected[0][0, :shared, 0], classified[:shared, 1])
    if mode == "circle":
        # QICK uses nearest-two even outside radius, and equality is ground.
        np.testing.assert_array_equal(classified[2:], [[0, 0, 1], [0, 0, 1]])


@pytest.mark.parametrize(
    "case",
    ["same_adc", "conflict", "missing", "centroid3", "empty", "mux", "mux_missing"],
)
def test_qick_options_cover_all_adcs_once_in_first_declaration_order(case):
    from tests.program.v2._mux_support import native_mux

    first = pulse(LinearClassifier(angle=0.75, threshold=0.25))
    repeated = first.model_copy(deep=True)
    mux = native_mux((4, 1))
    mux.tones[0].classifier = CircleClassifier(g_center=1 + 0j, e_center=2 + 0j)
    mux.tones[1].classifier = LinearClassifier(angle=-np.pi / 2, threshold=-2.0)
    configs = [first, mux, repeated]
    if case == "conflict":
        repeated.classifier = LinearClassifier(angle=0.75, threshold=99)
    elif case == "missing":
        repeated.classifier = None
    elif case == "centroid3":
        repeated.classifier = centroid3()
    elif case == "empty":
        configs = []
    elif case == "mux_missing":
        mux.tones[1].classifier = None
    options = readout.qick_threshold_options(mux if case.startswith("mux") else configs)
    if case in {"same_adc", "mux"}:
        np.testing.assert_allclose(
            options.get("threshold", []),
            [1.5, -2.0] if case == "mux" else [0.25, 1.5, -2.0],
        )
        np.testing.assert_allclose(
            options.get("angle", []),
            [0, np.pi / 2] if case == "mux" else [-0.75, 0, np.pi / 2],
        )
        assert options.get("remove_offset") is False
    else:
        assert options == {}


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


def test_centroid3_is_not_circle_calibration():
    with pytest.raises(ValueError, match="centroid3.*circle"):
        readout.circle_classifier(pulse(centroid3()), ro_ch=2)
