"""Readout calibration becomes explicit, per-ADC acquisition parameters."""

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


def test_circle_processor_preserves_reads_other_and_detached_adc_calibration():
    from tests.program.v2._mux_support import native_mux

    cfg = native_mux((4, 1))
    cfg.tones[0].classifier = CircleClassifier(g_center=0j, e_center=2 + 0j, radius=2.0)
    cfg.tones[1].classifier = CircleClassifier(g_center=0j, e_center=2j, radius=0.5)
    processors = readout.classification_processors(cfg, require_other=True)
    cfg.tones[0].classifier.g_center = 99j
    cfg.tones[0].classifier.radius = 0.1
    iq = np.array([[[0, 0], [2, 0]], [[1, 0], [0, 2]]], dtype=np.float64)
    original = iq.copy()
    assert list(processors) == [4, 1]
    result = processors[4](iq)
    np.testing.assert_array_equal(
        result, [[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]]]
    )
    assert result.dtype == np.float64
    np.testing.assert_array_equal(
        processors[1](iq), [[[1, 0, 0], [0, 0, 1]], [[0, 0, 1], [0, 1, 0]]]
    )
    np.testing.assert_array_equal(iq, original)


@pytest.mark.parametrize("mode", ["linear", "centroid3"])
def test_processor_emits_ordered_populations_with_ground_first_ties(mode):
    classifier = (
        LinearClassifier(angle=np.pi / 2, threshold=-0.5)
        if mode == "linear"
        else centroid3()
    )
    cfg = pulse(classifier)
    process = readout.classification_processors(cfg)[2]
    if isinstance(classifier, LinearClassifier):
        classifier.threshold = 100.0
        iq = np.array([[[0, 1], [0, -1], [0, 0.5]]], dtype=np.float64)
        expected = [[[1, 0], [0, 1], [1, 0]]]
    else:
        classifier.f_center = 99j
        iq = np.array(
            [[[0, 0], [2, 0], [0, 2]], [[1, 0], [0, 1], [2, 2]]],
            dtype=np.float64,
        )
        expected = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [1, 0, 0], [0, 1, 0]],
        ]
    result = process(iq)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64


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
    classified = readout.classification_processors(cfg)[0](iq)[:, 0]
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


def centroid3():
    return Centroid3Classifier(
        g_center=0j,
        e_center=2 + 0j,
        f_center=2j,
        confusion_matrix=[[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.0, 0.1, 0.9]],
    )


def test_centroid3_lookup_and_mapping_are_detached_and_use_adc_numbers():
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
    mapping = readout.classification_parameters(cfg)
    assert mapping == {4: first, 1: second}
    mapped = mapping[1]
    if not isinstance(mapped, Centroid3Classifier):
        pytest.fail("classification_parameters lost the centroid3 type")
    mapped.confusion_matrix[0][0] = -99.0
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


def test_centroid3_is_not_circle_other_or_herald_calibration():
    import numpy as np
    from zcu_tools.program.herald import HeraldSpec, herald_ground_mask

    cfg = pulse(centroid3())
    with pytest.raises(ValueError, match="centroid3.*circle"):
        readout.circle_classifier(cfg, ro_ch=2)
    with pytest.raises(ValueError, match="centroid3.*other"):
        readout.classification_parameters(cfg, require_other=True)
    with pytest.raises(ValueError, match="centroid3.*herald"):
        herald_ground_mask(np.array([0j], dtype=np.complex128), centroid3())
    with pytest.raises(ValueError, match="centroid3.*herald"):
        HeraldSpec(2, 0, centroid3(), 2)


@pytest.mark.parametrize(
    "classifier, require_other, message",
    [
        (None, False, "ADC 2: classifier missing"),
        (
            CircleClassifier(g_center=0j, e_center=1j),
            False,
            "ADC 2: circle radius.*Post",
        ),
        (LinearClassifier(angle=0.0, threshold=0.0), True, "ADC 2: linear.*other"),
        (centroid3(), True, "ADC 2: centroid3.*other"),
    ],
)
def test_processors_reject_unavailable_calibration(classifier, require_other, message):
    with pytest.raises(ValueError, match=message):
        readout.classification_processors(
            pulse(classifier), require_other=require_other
        )


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
