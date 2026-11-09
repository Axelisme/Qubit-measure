"""Continuous-block and thermal-shot moments through public processors."""

from contextlib import nullcontext

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import ThermalChainSums, thermal_chain_score
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program import v2
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseReadout

from tests.program._profiles import load_soccfg
from tests.program.v2._mux_support import mux_soccfg, native_mux


def block_readout(name: str = "readout") -> PulseReadout:
    return PulseReadout(
        name,
        PulseReadout.Cfg(
            gen_ch=0,
            ro_ch=0,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.1),
            ro_length=1.0,
        ),
    )


def block_program(n: int = 32) -> v2.TimelineProgramV2:
    builder = v2.TimelineBuilder(load_soccfg())
    with builder.repeat("block", n):
        builder.emit(block_readout(), tag="sample")
        builder.delay_auto()
    return finish_block(builder, n)


def finish_block(
    builder: v2.TimelineBuilder, declared: int = 32
) -> v2.TimelineProgramV2:
    prog = builder.build()
    prog.set_reads_per_shot(declared)
    return prog


# Scalar worked example: I=t, Q=2t+1 for t=0..31. Each lag k uses
# 32-k anchors. Order is means, second moments, all early-pair triples,
# then all late-pair triples, matching the research moment engine.
RAMP_MOMENTS = np.array(
    [
        31 / 2,
        32,
        651 / 2,
        1333 / 2,
        1365,
        # lag 1–15 II, IQ, QI, QQ
        320,
        655,
        656,
        1343,
        1885 / 6,
        3857 / 6,
        3869 / 6,
        3959 / 3,
        308,
        630,
        633,
        1295,
        603 / 2,
        1233 / 2,
        1241 / 2,
        1269,
        884 / 3,
        1807 / 3,
        1822 / 3,
        3725 / 3,
        575 / 2,
        1175 / 2,
        1187 / 2,
        1213,
        280,
        572,
        579,
        1183,
        1633 / 6,
        3335 / 6,
        3383 / 6,
        3455 / 3,
        264,
        539,
        548,
        1119,
        511 / 2,
        1043 / 2,
        1063 / 2,
        1085,
        740 / 3,
        1510 / 3,
        1543 / 3,
        3149 / 3,
        475 / 2,
        969 / 2,
        993 / 2,
        1013,
        228,
        465,
        478,
        975,
        1309 / 6,
        2669 / 6,
        2753 / 6,
        2807 / 3,
        208,
        424,
        439,
        895,
        # lag 1–4 I² I′, I² Q′, IQ I′, IQ Q′, Q² I′, Q² Q′
        7280,
        14865,
        14880,
        30385,
        30416,
        62113,
        41267 / 6,
        84245 / 6,
        84419 / 6,
        57449 / 2,
        57569 / 2,
        176306 / 3,
        6482,
        13230,
        13272,
        27090,
        27177,
        55475,
        6093,
        24867 / 2,
        24975 / 2,
        50967 / 2,
        51191 / 2,
        52236,
        # lag 1–4 I I′², I I′Q′, I Q′², Q I′², Q I′Q′, Q Q′²
        7600,
        15520,
        31695,
        15536,
        31728,
        64799,
        45037 / 6,
        30653 / 2,
        187775 / 6,
        30719 / 2,
        188183 / 6,
        192142 / 3,
        7406,
        15120,
        30870,
        15171,
        30975,
        63245,
        7299,
        29799 / 2,
        60831 / 2,
        29939 / 2,
        61119 / 2,
        62388,
    ],
    dtype=np.float64,
)


def test_block_sums_match_worked_113_moment_layout_without_cross_rep_pairs():
    process = v2.chain_readout_sums(block_program(), "sample")
    t = np.arange(32, dtype=np.float64)
    ramp = np.stack([t, 2 * t + 1], axis=-1)
    iq = np.stack([np.stack([ramp, 2 * ramp]), np.stack([-3 * ramp, 0 * ramp])])
    original = iq.copy()
    # Scaling both IQ components by a scales each degree-d moment by a**d.
    factors = np.array(
        [[3] * 2 + [5] * 63 + [9] * 48, [-3] * 2 + [9] * 63 + [-27] * 48]
    )
    expected = np.column_stack([np.full(2, 2), RAMP_MOMENTS * factors])

    actual = process({0: iq, 7: np.zeros((2, 2, 1, 2))})

    assert actual.shape == (2, 114)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    np.testing.assert_array_equal(iq, original)


@pytest.mark.parametrize(
    ("emitted", "declared"), [(31, 31), (33, 33), (32, 1), (32, 31), (32, 33)]
)
def test_block_requires_32_emitted_and_declared_reads(emitted: int, declared: int):
    prog = block_program(emitted)
    prog.set_reads_per_shot(declared)
    actual = emitted if emitted != 32 else declared
    with pytest.raises(ValueError, match=rf"ADC 0.*expected.*32.*actual.*{actual}"):
        v2.chain_readout_sums(prog, "sample")
    assert prog.ro_chs[0]["trigs"] == declared


def test_missing_label_reports_expected_actual_and_available_labels():
    with pytest.raises(
        ValueError, match=r"label 'absent'.*expected one.*actual 0.*sample"
    ):
        v2.chain_readout_sums(block_program(), "absent")


def test_label_matching_two_readout_owners_reports_owner_count():
    builder = v2.TimelineBuilder(load_soccfg())
    with builder.repeat("block", 16):
        builder.emit(block_readout("other"), tag="readout")
        builder.delay_auto()
        builder.emit(block_readout())
        builder.delay_auto()
    prog = finish_block(builder)
    with pytest.raises(
        ValueError, match=r"label 'readout'.*expected one.*actual 2.*readout"
    ):
        v2.chain_readout_sums(prog, "readout")


@pytest.mark.parametrize("unrolled", [False, True])
@pytest.mark.parametrize("tagged", [False, True])
@pytest.mark.parametrize("reps_innermost", [False, True])
def test_block_acquisition_adds_counts_per_point_over_reps_and_rounds(
    unrolled: bool, tagged: bool, reps_innermost: bool
):
    env = white_noise(load_soccfg(), std_adc=0, seed=7, routes=((0, 0),))
    try:
        builder = v2.TimelineBuilder(env.soccfg, reps=3, reps_innermost=reps_innermost)
        builder.add_loop("point", 2)
        readout = block_readout()
        with nullcontext() if unrolled else builder.repeat("block", 32):
            for index in range(32 if unrolled else 1):
                builder.emit(readout, tag="sample" if tagged and index == 0 else None)
                builder.delay_auto()
        prog = finish_block(builder)
        process = v2.chain_readout_sums(prog, "sample" if tagged else "readout")
        sums = prog.acquire_summed(env.soc, processor=process, rounds=2, progress=False)

        expected = np.zeros((2, 114), dtype=np.float64)
        expected[:, 0] = 6
        np.testing.assert_array_equal(sums, expected)
        assert prog.ro_chs[0]["trigs"] == 32
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize(
    "shape", [(1, 2, 31, 2), (1, 2, 33, 2), (1, 2, 32, 3), (1, 32, 2)]
)
def test_incoming_block_shape_reports_expected_and_actual(shape: tuple[int, ...]):
    process = v2.chain_readout_sums(block_program(), "sample")
    with pytest.raises(ValueError, match=r"ADC 0.*expected.*32, 2.*actual"):
        process({0: np.ones(shape, dtype=np.float64)})


@pytest.mark.parametrize("unrolled", [False, True])
def test_selected_adc_cannot_have_extra_triggers(unrolled: bool):
    builder = v2.TimelineBuilder(load_soccfg())
    readout = block_readout()
    with nullcontext() if unrolled else builder.repeat("block", 32):
        for _ in range(32 if unrolled else 1):
            builder.emit(readout)
            builder.delay_auto()
        builder.trigger([0])
        builder.delay_auto()
    expected, actual = (32, 33) if unrolled else (1, 2)
    with pytest.raises(
        ValueError, match=rf"ADC 0.*expected {expected}.*trigger.*actual {actual}"
    ):
        v2.chain_readout_sums(finish_block(builder), "readout")


@pytest.mark.parametrize("shape", ["nested", "register"])
def test_block_rejects_repeat_shapes_other_than_one_fixed_32(shape: str):
    builder = v2.TimelineBuilder(load_soccfg())
    readout = block_readout()
    if shape == "nested":
        with builder.repeat("outer", 2), builder.repeat("inner", 16):
            builder.emit(readout)
            builder.delay_auto()
        actual = r"2, 16"
    else:
        builder.add_loop("point", 1)
        builder.emit(v2.LoadValue("count", [32], idx_reg="point", val_reg="n_count"))
        with builder.repeat("block", "n_count", range_hint=(32, 32)):
            builder.emit(readout)
            builder.delay_auto()
        actual = "n_count"
    with pytest.raises(
        ValueError, match=rf"ADC 0.*expected 32.*fixed Repeat.*actual.*{actual}"
    ):
        v2.chain_readout_sums(finish_block(builder), "readout")


def test_unrolled_reads_cannot_be_conditional_on_a_branch():
    builder = v2.TimelineBuilder(load_soccfg())
    builder.add_loop("point", 2)
    readout = block_readout()
    with builder.branch(by="point", align_timeline=True) as cases, cases.case(0):
        for _ in range(32):
            builder.emit(readout)
            builder.timeline_inc(1.0)
    with pytest.raises(
        ValueError, match=r"ADC 0.*expected.*uncontrolled.*actual.*Branch"
    ):
        v2.chain_readout_sums(finish_block(builder), "readout")


def test_thermal_shot_columns_equal_shot_by_shot_random_iq_sums():
    process = v2.thermal_chain_sums(block_program(16), "sample")
    iq = np.random.default_rng(42).normal(size=(3, 23, 16, 2))
    original = iq.copy()
    expected = np.zeros((3, 38))
    for point, shots in enumerate(iq):
        for shot in shots:
            features = [*shot[[0, 1, 2, 3, 4, 14, 15]].ravel()]
            i, q = shot[0]
            features.extend([i * i, i * q, q * q])
            for early, late in ((0, 1), (0, 2), (0, 3), (0, 4), (14, 15)):
                features.extend(np.outer(shot[early], shot[late]).ravel())
            expected[point] += np.array([1, *features])

    actual = process({0: iq, 7: np.zeros((3, 23, 1, 2))})

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(iq, original)
    np.testing.assert_allclose(
        process({0: iq[:, :10]}) + process({0: iq[:, 10:]}), actual
    )


@pytest.mark.parametrize("unrolled", [False, True])
def test_thermal_acquisition_adds_shots_over_rounds(unrolled):
    env = white_noise(load_soccfg(), std_adc=0, seed=7, routes=((0, 0),))
    try:
        builder = v2.TimelineBuilder(env.soccfg, reps=3)
        builder.add_loop("point", 2)
        readout = block_readout()
        with nullcontext() if unrolled else builder.repeat("shot", 16):
            for _ in range(16 if unrolled else 1):
                builder.emit(readout)
                builder.delay_auto()
        prog = finish_block(builder, 16)
        sums = prog.acquire_summed(
            env.soc,
            processor=v2.thermal_chain_sums(prog, "readout"),
            rounds=2,
            progress=False,
        )
        expected = np.zeros((2, 38))
        expected[:, 0] = 6
        np.testing.assert_array_equal(sums, expected)
        assert prog.ro_chs[0]["trigs"] == 16
    finally:
        env.devices.close_all_devices()


@pytest.mark.parametrize("unrolled", [False, True])
def test_thermal_adc_rejects_other_triggers(unrolled):
    builder = v2.TimelineBuilder(load_soccfg())
    readout = block_readout()
    with nullcontext() if unrolled else builder.repeat("shot", 16):
        for _ in range(16 if unrolled else 1):
            builder.emit(readout)
            builder.delay_auto()
        builder.trigger([0])
        builder.delay_auto()
    expected, actual = (16, 17) if unrolled else (1, 2)
    with pytest.raises(
        ValueError, match=rf"ADC 0.*expected {expected}.*trigger.*actual {actual}"
    ):
        v2.thermal_chain_sums(finish_block(builder, 16), "readout")


@pytest.mark.parametrize(
    ("emitted", "declared"), [(15, 16), (17, 16), (16, 1), (16, 15), (16, 17)]
)
def test_thermal_shot_requires_16_emitted_and_declared_reads(emitted, declared):
    prog = block_program(emitted)
    prog.set_reads_per_shot(declared)
    actual = emitted if emitted != 16 else declared
    with pytest.raises(ValueError, match=rf"ADC 0.*expected.*16.*actual.*{actual}"):
        v2.thermal_chain_sums(prog, "sample")


@pytest.mark.parametrize(
    "shape", [(1, 2, 15, 2), (1, 2, 17, 2), (1, 2, 16, 3), (1, 16, 2)]
)
def test_thermal_shot_rejects_incoming_shape_mismatch(shape):
    process = v2.thermal_chain_sums(block_program(16), "sample")
    with pytest.raises(ValueError, match=r"ADC 0.*expected.*16, 2.*actual"):
        process({0: np.ones(shape)})


def test_thermal_unrolled_reads_reject_conditional_branch():
    builder = v2.TimelineBuilder(load_soccfg())
    builder.add_loop("point", 2)
    readout = block_readout()
    with builder.branch(by="point", align_timeline=True) as cases, cases.case(0):
        for _ in range(16):
            builder.emit(readout)
            builder.timeline_inc(1.0)
    with pytest.raises(
        ValueError, match=r"ADC 0.*expected.*uncontrolled.*actual.*Branch"
    ):
        v2.thermal_chain_sums(finish_block(builder, 16), "readout")


@pytest.mark.parametrize("shape", ["nested", "register"])
def test_thermal_rejects_nested_or_register_repeat(shape):
    builder = v2.TimelineBuilder(load_soccfg())
    readout = block_readout()
    if shape == "nested":
        with builder.repeat("outer", 2), builder.repeat("inner", 8):
            builder.emit(readout)
            builder.delay_auto()
        actual = r"2, 8"
    else:
        builder.add_loop("point", 1)
        builder.emit(v2.LoadValue("count", [16], idx_reg="point", val_reg="n_count"))
        with builder.repeat("shot", "n_count", range_hint=(16, 16)):
            builder.emit(readout)
            builder.delay_auto()
        actual = "n_count"
    with pytest.raises(
        ValueError, match=rf"ADC 0.*expected 16.*fixed Repeat.*actual.*{actual}"
    ):
        v2.thermal_chain_sums(finish_block(builder, 16), "readout")


def test_thermal_shot_iq_offset_leaves_saved_score_unchanged():
    process = v2.thermal_chain_sums(block_program(16), "sample")
    rng = np.random.default_rng(57)
    latent = rng.normal(size=(2, 1200, 1, 2))
    iq = latent + rng.normal(size=(2, 1200, 16, 2))
    axis = np.array([1.0, 2.0])
    baseline = thermal_chain_score(
        ThermalChainSums.from_processor(axis, process({0: iq}))
    )
    shifted = thermal_chain_score(
        ThermalChainSums.from_processor(axis, process({0: iq + [1.5, -2.5]}))
    )
    for original, offset in zip(baseline.settings, shifted.settings, strict=True):
        np.testing.assert_allclose(
            [
                offset.setting,
                offset.score,
                offset.score_error,
                offset.correlation,
                offset.qnd,
                offset.qnd_error,
                offset.mist,
            ],
            [
                original.setting,
                original.score,
                original.score_error,
                original.correlation,
                original.qnd,
                original.qnd_error,
                original.mist,
            ],
            rtol=1e-9,
        )
        assert offset.excluded == original.excluded
    assert shifted.ranking == baseline.ranking


def test_thermal_requires_label_to_identify_one_readout_owner():
    with pytest.raises(
        ValueError, match=r"label 'absent'.*expected one.*actual 0.*sample"
    ):
        v2.thermal_chain_sums(block_program(16), "absent")
    builder = v2.TimelineBuilder(load_soccfg())
    first, second = block_readout("first"), block_readout("sample")
    builder.emit(first, tag="sample")
    builder.delay_auto()
    builder.emit(second)
    builder.delay_auto()
    with pytest.raises(ValueError, match=r"label 'sample'.*expected one.*actual 2"):
        v2.thermal_chain_sums(finish_block(builder, 16), "sample")


def test_thermal_rejects_a_mux_readout_owner():
    builder = v2.TimelineBuilder(mux_soccfg())
    readout = native_mux().build("mux")
    with builder.repeat("shot", 16):
        builder.emit(readout)
        builder.delay_auto()
    with pytest.raises(ValueError, match=r"expected PulseReadout.*actual MuxReadout"):
        v2.thermal_chain_sums(finish_block(builder, 16), "mux")
