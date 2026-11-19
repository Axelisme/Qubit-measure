"""Read labels, classifiers and shot outer products through compiled programs."""

import numpy as np
import pytest
from qick.asm_v2 import QickSweep1D
from zcu_tools.cfg_model import Centroid3Classifier, CircleClassifier, LinearClassifier
from zcu_tools.program import v2
from zcu_tools.program.v2.modules import ConstWaveformCfg, Pulse, PulseReadout

from tests.program._profiles import load_soccfg
from tests.program._shot_reduction import reduce_round
from tests.program._timeline import compile_modules
from tests.program.v2._mux_support import mux_soccfg, native_mux
from tests.program.v2.timing_support import shots


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
    prog = compile_modules(load_soccfg(), [ro])
    iq = {
        0: np.array([[-1, 0], [0, 0], [1, 0], [3, 0]], dtype=np.float64).reshape(
            1, 4, 1, 2
        )
    }
    np.testing.assert_array_equal(
        reduce_round(v2.shot_sums(prog, [v2.Classify("readout")]), iq), [counts]
    )
    mixed = reduce_round(
        v2.shot_sums(prog, [v2.Classify("readout"), v2.IQ("readout")]), iq
    )
    expected_i = [[-1, 4], [-1, 1, 3], [-1, 1, 3]][
        ["linear", "circle", "centroid3"].index(classifier.mode)
    ]
    np.testing.assert_array_equal(mixed[0, :, 0], counts)
    np.testing.assert_array_equal(mixed[0, :, 1], expected_i)
    np.testing.assert_array_equal(mixed[0, :, 2], 0)
    reverse = reduce_round(
        v2.shot_sums(prog, [v2.IQ("readout"), v2.Classify("readout")]), iq
    )
    np.testing.assert_array_equal(reverse, mixed.transpose(0, 2, 1))
    # Explicit calibration overrides cfg, and the processor keeps a snapshot.
    override = LinearClassifier(angle=0.0, threshold=10.0)
    process = v2.shot_sums(prog, [v2.Classify("readout", classifier=override)])
    override.threshold = -10.0
    np.testing.assert_array_equal(reduce_round(process, iq), [[4, 0]])


def test_shot_sums_starts_with_zero_state_in_point_and_read_axis_order() -> None:
    prog = compile_modules(load_soccfg(), [readout()])
    reducer = v2.shot_sums(prog, [v2.IQ("readout", order=3), v2.IQ("readout")])

    state = reducer.start(3)

    assert state.dtype == np.float64
    np.testing.assert_array_equal(state, np.zeros((3, 10, 6)))


def test_iq_third_order_appends_cubic_sums_without_changing_second_order() -> None:
    prog = compile_modules(load_soccfg(), [readout()])
    iq = {0: np.array([[[[2.0, -3.0]], [[-1.0, 4.0]]]])}

    third = reduce_round(v2.shot_sums(prog, [v2.IQ("readout", order=3)]), iq)
    np.testing.assert_array_equal(third, [[2, 1, 1, 5, -10, 25, 7, -8, 2, 37]])

    default = reduce_round(v2.shot_sums(prog, [v2.IQ("readout")]), iq)
    expected = np.array([[2, 1, 1, 5, -10, 25]], dtype=np.float64)
    np.testing.assert_array_equal(default.view(np.uint64), expected.view(np.uint64))


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
    prog = compile_modules(load_soccfg(), [ro])
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
    result = reduce_round(process, {0: iq})
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(iq, original)


@pytest.mark.parametrize("swept", [False, True])
def test_branch_cases_share_read_labels_with_zero_gain_first_read(swept):
    r1_on = readout("r1")
    if swept:
        r1_on = PulseReadout("r1", r1_on.cfg, gain=QickSweep1D("gain", 0.1, 0.2))
    r1_off = PulseReadout("r1", r1_on.cfg, gain=0.0)
    r2 = PulseReadout("r2", r1_on.cfg)
    drive = Pulse(
        "drive",
        Pulse.Cfg(
            ch=1,
            nqz=1,
            freq=100.0,
            gain=0.2,
            waveform=ConstWaveformCfg(length=0.2),
        ),
    )
    builder = v2.TimelineBuilder(load_soccfg(), reps=2)
    if swept:
        builder.add_loop("gain", 2)
    builder.add_loop("branch", 8)
    with builder.branch(by="branch", align_timeline=True) as cases:
        for index in range(8):
            with cases.case(index):
                if index // 4:
                    builder.emit(drive)
                builder.emit(r1_on if index & 2 else r1_off)
                builder.timeline_inc(1.2)
                if index & 1:
                    builder.emit(drive)
                builder.emit(r2)
    prog = builder.build()
    process = v2.shot_sums(prog, [v2.IQ("r1"), v2.IQ("r2")])
    points = 16 if swept else 8
    iq = np.tile(np.array([[[[2.0, 3.0], [5.0, 7.0]]]]), (points, 2, 1, 1))
    result = reduce_round(process, {0: iq})
    assert result.shape == (points, 6, 6)
    np.testing.assert_array_equal(result[:, 0, 0], 2)
    np.testing.assert_array_equal(result[:, 3, 0], 8)
    np.testing.assert_array_equal(result[:, 0, 5], 98)
    np.testing.assert_array_equal(result[:, 1, 2], 28)
    starts = [
        [
            e.cycle - shot.start_cycle
            for e in shot.events
            if e.kind == "data" and e.value
        ]
        for shot in shots(prog)
    ]
    assert all(len(times) == 2 for times in starts)
    for group in range(points // 8):
        for off in (0, 1, 4, 5):
            assert starts[8 * group + off] == starts[8 * group + off + 2]
    assert prog.ro_chs[0]["trigs"] == 2


def _branch_program(case_modules):
    builder = v2.TimelineBuilder(load_soccfg(n_readouts=2))
    builder.add_loop("choice", len(case_modules))
    with builder.branch(by="choice", align_timeline=True) as cases:
        for index, modules in enumerate(case_modules):
            with cases.case(index):
                for module in modules:
                    builder.emit(module)
                    builder.timeline_inc(1.5)
    return builder.build()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ro_ch", 1),
        ("freq", 101.0),
        ("ro_length", 1.2),
        ("phase", 90.0),
        ("classifier", LinearClassifier(angle=0.0, threshold=1.0)),
    ],
)
def test_branch_adc_setting_mismatch_names_cases_and_field(field, value):
    first = readout("selected")
    cfg = first.cfg.model_copy(update={field: value})
    other = PulseReadout("selected", cfg)
    if field == "ro_ch":
        # Keep one read per ADC in each case; only the label changes ADC.
        case_modules = [[first, readout("pad1", ch=1)], [other, readout("pad0")]]
    else:
        case_modules = [[first], [other]]
    prog = _branch_program(case_modules)
    with pytest.raises(ValueError, match=rf"choice\[0\].*choice\[1\].*{field}"):
        v2.shot_sums(prog, [v2.IQ("selected")])


def test_branch_read_index_mismatch_is_rejected_even_with_equal_counts():
    selected, auxiliary = readout("selected"), readout("auxiliary")
    prog = _branch_program([[selected, auxiliary], [auxiliary, selected]])
    with pytest.raises(ValueError, match=r"choice\[0\].*choice\[1\].*read index"):
        v2.shot_sums(prog, [v2.IQ("selected")])


def test_branch_generator_waveform_can_differ_with_identical_adc_settings():
    first = readout("selected")
    other = PulseReadout("selected", first.cfg, gain=0.0, length=0.2)
    prog = _branch_program([[first], [other]])
    result = reduce_round(
        v2.shot_sums(prog, [v2.IQ("selected")]),
        {0: np.array([[[[2.0, 3.0]]], [[[5.0, 7.0]]]])},
    )
    np.testing.assert_array_equal(result, [[1, 2, 3, 4, 6, 9], [1, 5, 7, 25, 35, 49]])


@pytest.mark.parametrize("timeline", [True])
@pytest.mark.parametrize("mux", [False, True])
def test_labels_resolve_adc_trigger_order_and_repeated_emissions(timeline, mux):
    cfg = mux_soccfg() if mux else load_soccfg()
    ro = native_mux((4, 1)).build("readout") if mux else readout()

    class Program(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(ro)

        def _body(self, cfg):
            self.emit(ro, tag="prefix")
            # Each pulse ADC integrates for 1 us; generator completion is earlier.
            self.timeline_inc(1.0)
            self.emit(ro, tag="herald")
            self.timeline_inc(1.0)
            self.emit(ro, tag="signal")

    prog = Program(cfg, reps=2)
    names = ["herald", "signal"]
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
    result = reduce_round(v2.shot_sums(prog, axes), iq)
    herald = np.array([[1, 2, 3, 4, 6, 9], [1, 11, 13, 121, 143, 169]])
    signal = np.array([[1, 5, 7, 25, 35, 49], [1, 17, 19, 289, 323, 361]])
    expected = (herald.T @ signal)[None, ...]
    np.testing.assert_array_equal(result, expected)


def test_shot_sums_resolves_labels_after_mask_sources_remove_physical_reads() -> None:
    builder = v2.TimelineBuilder(load_soccfg(), reps=2)
    ro = readout()
    builder.emit(ro, tag="reference")
    builder.timeline_inc(1.0)
    builder.emit(ro, tag="mask_input")
    builder.timeline_inc(1.0)
    builder.emit(ro, tag="signal")
    builder.register_shot_mask_source({0: [1]}, lambda iq: iq[0][..., 0, 0] > 0)
    program = builder.build()

    sums = v2.shot_sums(program, [v2.IQ("signal")])
    logical_iq = {
        0: np.array([[[[99.0, 99.0], [2.0, 3.0]], [[98.0, 98.0], [5.0, 7.0]]]])
    }
    np.testing.assert_array_equal(
        reduce_round(sums, logical_iq), [[2, 7, 10, 29, 41, 58]]
    )
    with pytest.raises(
        ValueError, match=r"label 'mask_input'.*available labels.*signal"
    ):
        v2.shot_sums(program, [v2.IQ("mask_input")])


@pytest.mark.parametrize("timeline", [True])
def test_empty_axes_fail_when_constructing_processor(timeline):
    prog = (
        v2.TimelineProgramV2(load_soccfg())
        if timeline
        else compile_modules(load_soccfg(), [readout()])
    )
    with pytest.raises(ValueError, match="axes must contain at least one read"):
        v2.shot_sums(prog, [])


@pytest.mark.parametrize("reads", [0, 2])
def test_processor_rejects_adc_read_count_mismatch(reads):
    prog = compile_modules(load_soccfg(), [readout()])
    process = v2.shot_sums(prog, [v2.IQ("readout")])
    with pytest.raises(ValueError, match=rf"ADC 0.*expected.*1.*actual.*{reads}"):
        reduce_round(process, {0: np.zeros((1, 2, reads, 2), dtype=np.float64)})


@pytest.mark.parametrize(
    ("mux", "tone"), [(False, 0), (True, None), (True, -1), (True, 2), (True, True)]
)
def test_tone_must_match_readout_type(mux, tone):
    ro = native_mux((4, 1)).build("readout") if mux else readout()
    prog = compile_modules(mux_soccfg() if mux else load_soccfg(), [ro])
    with pytest.raises(ValueError, match="tone"):
        v2.shot_sums(prog, [v2.IQ("readout", tone=tone)])


@pytest.mark.parametrize("inside", [False, True])
def test_repeated_triggers_on_selected_adc_are_rejected(inside):
    ro = readout()

    class Program(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(ro)

        def _body(self, cfg):
            if not inside:
                self.emit(ro, tag="selected")
                # Clear the 1-us ADC window before entering controlled triggers.
                self.timeline_inc(1.0)
            self.delay_by_timeline()
            with self.repeat("repeat_scope", 2):
                if inside:
                    self.emit(ro, tag="selected")
                else:
                    self.trigger([0], t=0, timeline="default")
                self.timeline_inc(1.0)
                self.delay_by_timeline()

    prog = Program(load_soccfg())
    with pytest.raises(ValueError, match=r"ADC 0.*Repeat"):
        v2.shot_sums(prog, [v2.IQ("selected")])


@pytest.mark.parametrize("control", ["Repeat", "Branch"])
def test_control_flow_on_other_adc_preserves_selected_read_sums(control):
    selected = readout()
    other = readout("other", ch=1)
    builder = v2.TimelineBuilder(load_soccfg(n_readouts=2))
    builder.add_loop("choice", 2)
    if control == "Repeat":
        with builder.repeat("repeat_scope", 2):
            builder.emit(other)
            # ADC 1 integrates for 1 us independently of the selected ADC 0.
            builder.timeline_inc(1.0)
    else:
        with builder.branch(by="choice", align_timeline=True) as cases:
            for index in range(2):
                with cases.case(index):
                    builder.emit(other)
    builder.emit(selected)
    prog = builder.build()
    process = v2.shot_sums(prog, [v2.IQ("readout")])
    result = reduce_round(
        process,
        {
            0: np.array([[[[4.0, 5.0]]]]),
            1: np.zeros((1, 1, 2 if control == "Repeat" else 1, 2), dtype=np.float64),
        },
    )
    np.testing.assert_array_equal(result, [[1, 4, 5, 16, 20, 25]])


@pytest.mark.parametrize("inner", ["Repeat", "Branch"])
def test_nested_branch_and_repeat_still_reject_repeated_adc_trigger(inner):
    ro = readout()
    builder = v2.TimelineBuilder(load_soccfg())
    builder.add_loop("choice", 2)
    builder.emit(ro, tag="selected")
    # Separate the selected 1-us window from nested reads on the same ADC.
    builder.timeline_inc(1.0)
    if inner == "Repeat":
        with builder.branch(by="choice", align_timeline=True) as cases:
            for index in range(2):
                with cases.case(index), builder.repeat(f"reads_{index}", 2):
                    builder.emit(ro)
                    builder.timeline_inc(1.0)
    else:
        with builder.repeat("reads", 2):
            with builder.branch(by="choice", align_timeline=True) as cases:
                for index in range(2):
                    with cases.case(index):
                        builder.emit(ro)
                        builder.timeline_inc(1.0)
            # Clear each iteration before reentering the conditional read.
            builder.delay_by_timeline()
    prog = builder.build()
    with pytest.raises(ValueError, match=r"Branch.*choice\[0\].*ADC 0.*Repeat"):
        v2.shot_sums(prog, [v2.IQ("selected")])


def test_split_sums_returns_counts_and_nan_means_without_warnings():
    sums = np.array(
        [
            [[2, 6, -4, 20, -12, 8], [0, 0, 0, 0, 0, 0]],
            [[1, -3, 7, 9, -21, 49], [np.nan] * 6],
        ],
        dtype=np.float64,
    )
    with np.errstate(all="raise"):
        counts, means = v2.split_sums(sums)
    np.testing.assert_array_equal(counts, [[2, 0], [1, np.nan]])
    np.testing.assert_array_equal(
        means, [[3 - 2j, complex(np.nan, np.nan)], [-3 + 7j, complex(np.nan, np.nan)]]
    )
    assert means.dtype == np.complex128
    for length in (2, 3, 5, 7):
        with pytest.raises(ValueError, match=f"6.*{length}"):
            v2.split_sums(np.zeros((4, length), dtype=np.float64))
    with pytest.raises(ValueError, match="6.*\\(\\)"):
        v2.split_sums(np.array(1.0))


@pytest.mark.parametrize("timeline", [True])
@pytest.mark.parametrize("collision", [False, True])
def test_missing_and_ambiguous_labels_list_available_labels(timeline, collision):
    ro = readout()

    class Program(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(ro)

        def _body(self, cfg):
            self.emit(ro, tag="readout" if collision else None)
            # Reach label validation after two nonoverlapping 1-us ADC reads.
            self.timeline_inc(1.0)
            self.emit(ro)

    prog = Program(load_soccfg())
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
        Program(load_soccfg())


@pytest.mark.parametrize(
    "classifier", [None, CircleClassifier(g_center=0j, e_center=2 + 0j)]
)
def test_missing_or_incomplete_classification_fails_at_construction(classifier):
    prog = compile_modules(load_soccfg(), [readout(classifier=classifier)])
    with pytest.raises(ValueError, match="classifier missing|GE Post"):
        v2.shot_sums(prog, [v2.Classify("readout")])


def test_parallel_readout_emissions_keep_labels_without_changing_admission():
    first = readout("first", ch=0)
    second = readout("second", ch=1)

    class Program(v2.TimelineProgramV2):
        def _initialize(self, cfg):
            self.declare(first)
            self.declare(second)

        def _body(self, cfg):
            self.fork("default", into="second")
            self.emit(first)
            self.emit(second, timeline="second")
            self.join("default", "second", into="default")

    prog = Program(load_soccfg(n_gens=2, n_readouts=2))
    process = v2.shot_sums(prog, [v2.IQ("first"), v2.IQ("second")])
    iq = {ch: np.full((1, 2, 1, 2), ch + 1.0) for ch in (0, 1)}
    expected = 2 * np.outer([1, 1, 1, 1, 1, 1], [1, 2, 2, 4, 4, 4])
    np.testing.assert_array_equal(reduce_round(process, iq), expected[None, ...])


@pytest.mark.parametrize("tone", [None, 0])
def test_single_tone_mux_classify_resolves_implicit_and_explicit_tone(tone):
    cfg = native_mux((4,))
    cfg.tones[0].classifier = LinearClassifier(angle=0.0, threshold=1.0)
    prog = compile_modules(mux_soccfg(), [cfg.build("readout")])
    iq = {
        4: np.array([[-1, 0], [0, 0], [1, 0], [2, 0]], dtype=np.float64).reshape(
            1, 4, 1, 2
        )
    }
    process = v2.shot_sums(prog, [v2.Classify("readout", tone=tone)])
    np.testing.assert_array_equal(reduce_round(process, iq), [[3, 1]])


def test_single_tone_mux_iq_resolves_implicit_and_explicit_tone():
    cfg = native_mux((4,))
    prog = compile_modules(mux_soccfg(), [cfg.build("readout")])
    iq = {
        cfg.tones[0].ro_ch: np.array(
            [[[[2, 3]], [[4, 5]]], [[[-1, 2]], [[1, -2]]]], dtype=np.float64
        )
    }
    implicit = reduce_round(v2.shot_sums(prog, [v2.IQ("readout")]), iq)
    explicit = reduce_round(v2.shot_sums(prog, [v2.IQ("readout", 0)]), iq)
    np.testing.assert_array_equal(
        implicit, [[2, 6, 8, 20, 26, 34], [2, 0, 0, 2, -4, 8]]
    )
    np.testing.assert_array_equal(implicit, explicit)


@pytest.mark.parametrize(
    "axis",
    [
        v2.IQ("readout"),
        v2.Classify("readout", classifier=LinearClassifier(angle=0.0, threshold=0.0)),
    ],
)
def test_multi_tone_mux_requires_tone_and_lists_available_positions(axis):
    prog = compile_modules(mux_soccfg(), [native_mux((4, 1)).build("readout")])
    with pytest.raises(ValueError, match=r"readout.*available tones.*0.*1"):
        v2.shot_sums(prog, [axis])


def test_mux_classify_only_axes_produce_joint_counts():
    cfg = native_mux((4, 1))
    for tone in cfg.tones:
        tone.classifier = LinearClassifier(angle=0.0, threshold=0.0)
    prog = compile_modules(mux_soccfg(), [cfg.build("readout")])
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
    np.testing.assert_array_equal(reduce_round(process, iq), [[[1, 2], [1, 0]]])


def test_iq_axis_sums_and_repeated_read_second_order_features():
    prog = compile_modules(load_soccfg(), [readout()])
    iq = {0: np.array([[[[2, 3]], [[4, 5]]], [[[-1, 2]], [[1, -2]]]], dtype=np.float64)}
    features = np.array(
        [
            [[1, 2, 3, 4, 6, 9], [1, 4, 5, 16, 20, 25]],
            [[1, -1, 2, 1, -2, 4], [1, 1, -2, 1, -2, 4]],
        ]
    )
    np.testing.assert_array_equal(
        reduce_round(v2.shot_sums(prog, [v2.IQ("readout")]), iq),
        [[2, 6, 8, 20, 26, 34], [2, 0, 0, 2, -4, 8]],
    )
    actual = reduce_round(v2.shot_sums(prog, [v2.IQ("readout"), v2.IQ("readout")]), iq)
    expected = np.stack([point.T @ point for point in features])
    assert actual.shape == (2, 6, 6)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


def test_shot_sums_rejects_read_within_a_forward_jump_scope():
    builder = v2.TimelineBuilder(load_soccfg())
    builder.add_loop("depth", 2)
    builder.cond_jump("alternate", "depth", "Z")
    builder.emit(readout(), tag="first")
    builder.jump("join")
    builder.label("alternate")
    builder.emit(readout(), tag="second")
    builder.label("join")

    error = "ADC 0: trigger inside Jump 'join' is unsupported for read labels"
    with pytest.raises(ValueError, match=f"^{error}$"):
        v2.shot_sums(builder.build(), [v2.IQ("first")])


def test_shot_sums_resolves_shared_read_after_the_jump_target():
    builder = v2.TimelineBuilder(load_soccfg())
    builder.add_loop("depth", 2)
    builder.cond_jump("read", "depth", "Z")
    builder.timeline_inc(0.2)
    builder.label("read")
    builder.emit(readout(), tag="shared")

    reducer = v2.shot_sums(builder.build(), [v2.IQ("shared")])
    iq = np.array([[[[2.0, 3.0]]], [[[4.0, 5.0]]]])
    np.testing.assert_array_equal(
        reduce_round(reducer, {0: iq}), [[1, 2, 3, 4, 6, 9], [1, 4, 5, 16, 20, 25]]
    )
