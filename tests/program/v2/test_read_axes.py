"""Read labels, classifiers and shot outer products through compiled programs."""

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import Centroid3Classifier, CircleClassifier, LinearClassifier
from zcu_tools.program import v2
from zcu_tools.program.v2.modules import (
    Branch,
    ConstWaveformCfg,
    Join,
    Module,
    PulseReadout,
    Repeat,
)

from tests.program.v2._mux_support import mux_soccfg, native_mux


def readout(name="readout", classifier=None, ch=0):
    return PulseReadout(
        name,
        PulseReadout.Cfg(
            gen_ch=ch,
            ro_ch=ch,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=1.0,
            classifier=classifier,
        ),
    )


@pytest.mark.parametrize(
    ("classifier", "counts"),
    [
        (LinearClassifier(angle=0.0, threshold=0.0), [2, 2]),
        (CircleClassifier(g_center=-1 + 0j, e_center=1 + 0j, radius=1.0), [1, 1, 2]),
        (
            Centroid3Classifier(
                g_center=-1 + 0j,
                e_center=1 + 0j,
                f_center=3 + 0j,
                confusion_matrix=np.eye(3).tolist(),
            ),
            [2, 1, 1],
        ),
    ],
)
def test_classified_and_mixed_axes_preserve_class_rules_and_axis_order(
    classifier, counts
):
    ro = readout(classifier=classifier)
    prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [ro])
    iq = {
        0: np.array([[-1, 0], [0, 0], [1, 0], [3, 0]], dtype=np.float64).reshape(
            1, 4, 1, 2
        )
    }
    np.testing.assert_array_equal(
        v2.shot_sums(prog, [v2.Classify("readout")])(iq), [counts]
    )
    mixed = v2.shot_sums(prog, [v2.Classify("readout"), v2.IQ("readout")])(iq)
    expected_i = [[-1, 4], [-1, 1, 3], [-1, 1, 3]][
        ["linear", "circle", "centroid3"].index(classifier.mode)
    ]
    np.testing.assert_array_equal(mixed[0, :, 0], counts)
    np.testing.assert_array_equal(mixed[0, :, 1], expected_i)
    np.testing.assert_array_equal(mixed[0, :, 2], 0)
    reverse = v2.shot_sums(prog, [v2.IQ("readout"), v2.Classify("readout")])(iq)
    np.testing.assert_array_equal(reverse, mixed.transpose(0, 2, 1))
    # Explicit calibration overrides cfg, and the processor keeps a snapshot.
    override = LinearClassifier(angle=0.0, threshold=10.0)
    process = v2.shot_sums(prog, [v2.Classify("readout", classifier=override)])
    override.threshold = -10.0
    np.testing.assert_array_equal(process(iq), [[4, 0]])


@pytest.mark.parametrize(
    ("classifier", "values", "expected"),
    [
        (
            CircleClassifier(g_center=0j, e_center=2 + 0j, radius=2.0),
            [[0, 0], [2, 0], [1, 0], [0, 2]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        ),
        (
            LinearClassifier(angle=np.pi / 2, threshold=-0.5),
            [[0, 1], [0, -1], [0, 0.5]],
            [[1, 0], [0, 1], [1, 0]],
        ),
        (
            Centroid3Classifier(
                g_center=0j,
                e_center=2 + 0j,
                f_center=2j,
                confusion_matrix=np.eye(3).tolist(),
            ),
            [[0, 0], [2, 0], [0, 2], [1, 0], [0, 1], [2, 2]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]],
        ),
    ],
)
def test_classified_sums_keep_boundary_rules_and_detached_cfg(
    classifier, values, expected
):
    ro = readout(classifier=classifier)
    prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [ro])
    process = v2.shot_sums(prog, [v2.Classify("readout")])
    if isinstance(ro.cfg.classifier, CircleClassifier):
        ro.cfg.classifier.g_center = 99j
        ro.cfg.classifier.radius = 0.1
    elif isinstance(ro.cfg.classifier, LinearClassifier):
        ro.cfg.classifier.threshold = 100.0
    elif isinstance(ro.cfg.classifier, Centroid3Classifier):
        ro.cfg.classifier.f_center = 99j
    iq = np.array(values, dtype=np.float64).reshape(-1, 1, 1, 2)
    original = iq.copy()
    result = process({0: iq})
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(iq, original)


@pytest.mark.parametrize("timeline", [False, True])
@pytest.mark.parametrize("mux", [False, True])
def test_labels_resolve_adc_trigger_order_and_repeated_emissions(timeline, mux):
    cfg = mux_soccfg() if mux else v2.make_mock_soccfg()
    ro = native_mux((4, 1)).build("readout") if mux else readout()
    if timeline:

        class Program(v2.TimelineProgramV2):
            def _initialize(self, cfg):
                self.declare(ro)

            def _body(self, cfg):
                self.emit(ro, tag="prefix")
                self.emit(ro, tag="herald")
                self.emit(ro, tag="signal")

        prog = Program(cfg, reps=2)
        names = ["herald", "signal"]
    else:

        class Reads(Module):
            name = "reads"

            def init(self, prog):
                ro.init(prog)

            def run(self, prog, t: float | QickParam = 0.0):
                channels = [4, 1] if mux else [0]
                assert t == 0.0
                prog.trigger(channels, t=0)
                ro.run(prog, t=2.0)
                prog.trigger(channels, t=4)
                return t + 6.0

        prog = v2.ModularProgramV2(cfg, [Reads()], reps=2)
        names = ["readout", "readout"]
    axes = [v2.IQ(name, tone=1 if mux else None) for name in names]
    iq = {
        ch: np.array(
            [[[[99, 99], [2, 3], [5, 7]], [[99, 99], [11, 13], [17, 19]]]],
            dtype=np.float64,
        )
        for ch in prog.ro_chs
    }
    if mux:
        iq[4] *= -1  # The requested tone position 1 is ADC 1, not first ADC 4.
    result = v2.shot_sums(prog, axes)(iq)
    expected = (
        [[[2, 22, 26], [13, 197, 223], [16, 236, 268]]]
        if timeline
        else [[[2, 13, 16], [13, 125, 149], [16, 149, 178]]]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("timeline", [False, True])
def test_empty_axes_fail_when_constructing_processor(timeline):
    prog = (
        v2.TimelineProgramV2(v2.make_mock_soccfg())
        if timeline
        else v2.ModularProgramV2(v2.make_mock_soccfg(), [readout()])
    )
    with pytest.raises(ValueError, match="axes must contain at least one read"):
        v2.shot_sums(prog, [])


@pytest.mark.parametrize("reads", [0, 2])
def test_processor_rejects_adc_read_count_mismatch(reads):
    prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [readout()])
    process = v2.shot_sums(prog, [v2.IQ("readout")])
    with pytest.raises(ValueError, match=rf"ADC 0.*expected.*1.*actual.*{reads}"):
        process({0: np.zeros((1, 2, reads, 2), dtype=np.float64)})


@pytest.mark.parametrize(
    ("mux", "tone"), [(False, 0), (True, None), (True, -1), (True, 2), (True, True)]
)
def test_tone_must_match_readout_type(mux, tone):
    ro = native_mux((4, 1)).build("readout") if mux else readout()
    prog = v2.ModularProgramV2(mux_soccfg() if mux else v2.make_mock_soccfg(), [ro])
    with pytest.raises(ValueError, match="tone"):
        v2.shot_sums(prog, [v2.IQ("readout", tone=tone)])


def test_declared_reads_must_match_actual_triggers():
    prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [readout()])
    prog.set_reads_per_shot(2)
    with pytest.raises(ValueError, match="ADC 0.*2.*1"):
        v2.shot_sums(prog, [v2.IQ("readout")])


@pytest.mark.parametrize("timeline", [False, True])
@pytest.mark.parametrize("control", ["Repeat", "Branch"])
@pytest.mark.parametrize("inside", [False, True])
def test_any_control_flow_trigger_on_selected_adc_is_rejected(
    timeline, control, inside
):
    ro = readout()
    if timeline:

        class Program(v2.TimelineProgramV2):
            def _initialize(self, cfg):
                self.declare(ro)
                self.add_loop("choice", 2)

            def _body(self, cfg):
                if not inside:
                    self.emit(ro, tag="selected")
                if control == "Repeat":
                    self.sync()
                    with self.repeat("repeat_scope", 2):
                        if inside:
                            self.emit(ro, tag="selected")
                        else:
                            self.trigger([0], t=0, timeline="default")
                        self.sync()
                else:
                    with self.branch(by="choice", align_timeline=True) as branch:
                        with branch.case(0):
                            if inside:
                                self.emit(ro, tag="selected")
                            else:
                                self.trigger([0], t=0)
                        with branch.case(1):
                            pass

        prog = Program(v2.make_mock_soccfg())
        label = "selected"
    else:
        target = ro if inside else readout("other")
        scope = (
            Repeat("repeat_scope", 2).add_content(target)
            if control == "Repeat"
            else Branch("choice", target, None)
        )
        prog = v2.ModularProgramV2(
            v2.make_mock_soccfg(),
            [scope] if inside else [ro, scope],
            sweep=[("choice", 2)],
        )
        label = "readout"
    with pytest.raises(ValueError, match=rf"ADC 0.*{control}"):
        v2.shot_sums(prog, [v2.IQ(label)])


@pytest.mark.parametrize("control", ["Repeat", "Branch"])
def test_control_flow_on_other_adc_preserves_selected_read_sums(control):
    selected = readout()
    other = readout("other", ch=1)
    scope = (
        Repeat("repeat_scope", 2).add_content(other)
        if control == "Repeat"
        else Branch("choice", other, None)
    )
    prog = v2.ModularProgramV2(
        v2.make_mock_soccfg(n_readouts=2),
        [scope, selected],
        sweep=[("choice", 2)],
    )
    process = v2.shot_sums(prog, [v2.IQ("readout")])
    result = process(
        {
            0: np.array([[[[4.0, 5.0]]]]),
            1: np.zeros((1, 1, 1, 2), dtype=np.float64),
        }
    )
    np.testing.assert_array_equal(result, [[1, 4, 5]])


def test_split_sums_returns_counts_and_nan_means_without_warnings():
    sums = np.array(
        [[[2, 6, -4], [0, 0, 0]], [[1, -3, 7], [np.nan, np.nan, np.nan]]],
        dtype=np.float64,
    )
    with np.errstate(all="raise"):
        counts, means = v2.split_sums(sums)
    np.testing.assert_array_equal(counts, [[2, 0], [1, np.nan]])
    np.testing.assert_array_equal(
        means, [[3 - 2j, complex(np.nan, np.nan)], [-3 + 7j, complex(np.nan, np.nan)]]
    )
    assert means.dtype == np.complex128
    with pytest.raises(ValueError, match="3.*2"):
        v2.split_sums(np.zeros((4, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="3.*\\(\\)"):
        v2.split_sums(np.array(1.0))


@pytest.mark.parametrize("timeline", [False, True])
@pytest.mark.parametrize("collision", [False, True])
def test_missing_and_ambiguous_labels_list_available_labels(timeline, collision):
    ro = readout()
    if timeline:

        class Program(v2.TimelineProgramV2):
            def _initialize(self, cfg):
                self.declare(ro)

            def _body(self, cfg):
                self.emit(ro, tag="readout" if collision else None)
                self.emit(ro)

        prog = Program(v2.make_mock_soccfg())
    else:

        class Reads(Module):
            name = "reads"

            def init(self, prog):
                ro.init(prog)

            def run(self, prog, t=0.0):
                return ro.run(prog, ro.run(prog, t))

        prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [Reads()])
    label = "readout" if collision else "missing"
    with pytest.raises(ValueError, match=rf"{label}.*available labels.*readout"):
        v2.shot_sums(prog, [v2.IQ(label)])


@pytest.mark.parametrize("native_conflict", [False, True])
def test_explicit_tags_keep_qick_compile_time_uniqueness(native_conflict):
    ro = readout()

    class Program(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(ro)

        def _body(self, cfg):
            self.emit(ro, tag="duplicate")
            if native_conflict:
                self.delay(1.0, tag="duplicate")
            else:
                self.emit(ro, tag="duplicate")

    with pytest.raises(
        RuntimeError, match="two instructions have the same tag duplicate"
    ):
        Program(v2.make_mock_soccfg())


@pytest.mark.parametrize(
    "classifier", [None, CircleClassifier(g_center=0j, e_center=2 + 0j)]
)
def test_missing_or_incomplete_classification_fails_at_construction(classifier):
    prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [readout(classifier=classifier)])
    with pytest.raises(ValueError, match="classifier missing|GE Post"):
        v2.shot_sums(prog, [v2.Classify("readout")])


def test_parallel_readout_emissions_keep_labels_without_changing_admission():
    first = readout("first", ch=0)
    second = readout("second", ch=1)
    parallel = Join(first, second)

    class Program(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(parallel)

        def _body(self, cfg):
            self.emit(parallel)

    prog = Program(v2.make_mock_soccfg(n_gens=2, n_readouts=2))
    process = v2.shot_sums(prog, [v2.IQ("first"), v2.IQ("second")])
    iq = {ch: np.full((1, 2, 1, 2), ch + 1.0) for ch in (0, 1)}
    np.testing.assert_array_equal(process(iq), [[[2, 4, 4], [2, 4, 4], [2, 4, 4]]])


@pytest.mark.parametrize("tone", [None, 0])
def test_single_tone_mux_classify_resolves_implicit_and_explicit_tone(tone):
    cfg = native_mux((4,))
    cfg.tones[0].classifier = LinearClassifier(angle=0.0, threshold=1.0)
    prog = v2.ModularProgramV2(mux_soccfg(), [cfg.build("readout")])
    iq = {
        4: np.array([[-1, 0], [0, 0], [1, 0], [2, 0]], dtype=np.float64).reshape(
            1, 4, 1, 2
        )
    }
    process = v2.shot_sums(prog, [v2.Classify("readout", tone=tone)])
    np.testing.assert_array_equal(process(iq), [[3, 1]])


def test_multi_tone_mux_classify_requires_tone_and_lists_available_positions():
    prog = v2.ModularProgramV2(mux_soccfg(), [native_mux((4, 1)).build("readout")])
    with pytest.raises(ValueError, match=r"readout.*available tones.*0.*1"):
        v2.shot_sums(
            prog,
            [
                v2.Classify(
                    "readout", classifier=LinearClassifier(angle=0.0, threshold=0.0)
                )
            ],
        )


def test_mux_classify_only_axes_produce_joint_counts():
    cfg = native_mux((4, 1))
    for tone in cfg.tones:
        tone.classifier = LinearClassifier(angle=0.0, threshold=0.0)
    prog = v2.ModularProgramV2(mux_soccfg(), [cfg.build("readout")])
    iq = {
        4: np.array([[-1, 0], [-1, 0], [1, 0], [-1, 0]], dtype=np.float64).reshape(
            1, 4, 1, 2
        ),
        1: np.array([[-1, 0], [1, 0], [-1, 0], [1, 0]], dtype=np.float64).reshape(
            1, 4, 1, 2
        ),
    }
    process = v2.shot_sums(
        prog, [v2.Classify("readout", tone=0), v2.Classify("readout", tone=1)]
    )
    np.testing.assert_array_equal(process(iq), [[[1, 2], [1, 0]]])


def test_iq_axis_sums_and_repeated_read_second_moments():
    prog = v2.ModularProgramV2(v2.make_mock_soccfg(), [readout()])
    iq = {0: np.array([[[[2, 3]], [[4, 5]]], [[[-1, 2]], [[1, -2]]]], dtype=np.float64)}
    process = v2.shot_sums(prog, [v2.IQ("readout"), v2.IQ("readout")])
    actual = process(iq)
    np.testing.assert_array_equal(
        actual,
        [
            [[2, 6, 8], [6, 20, 26], [8, 26, 34]],
            [[2, 0, 0], [0, 2, -4], [0, -4, 8]],
        ],
    )
    np.testing.assert_array_equal(
        v2.shot_sums(prog, [v2.IQ("readout")])(iq), [[2, 6, 8], [2, 0, 0]]
    )
    assert actual.dtype == np.float64
