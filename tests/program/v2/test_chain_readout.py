"""Thermal-shot moments through public processors."""

from contextlib import nullcontext

import numpy as np
import pytest
from zcu_tools.analysis.readout_chain import ThermalChainSums, thermal_chain_score
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program import v2
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseReadout

from tests.program._profiles import load_soccfg
from tests.program._shot_reduction import reduce_round
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


def block_program(n: int = 16) -> v2.TimelineProgramV2:
    builder = v2.TimelineBuilder(load_soccfg())
    with builder.repeat("block", n):
        builder.emit(block_readout(), tag="sample")
        builder.delay_auto()
    return finish_block(builder, n)


def finish_block(
    builder: v2.TimelineBuilder, declared: int = 16
) -> v2.TimelineProgramV2:
    prog = builder.build()
    prog.set_reads_per_shot(declared)
    return prog


def test_thermal_shot_columns_equal_shot_by_shot_random_iq_sums():
    process = v2.thermal_chain_sums(block_program(16), "sample")
    iq = np.random.default_rng(42).normal(size=(3, 23, 16, 2))
    original = iq.copy()
    expected = np.zeros((3, 108))
    for point, shots in enumerate(iq):
        for shot in shots:
            features = [*shot.ravel()]
            i, q = shot[0]
            features.extend([i * i, i * q, q * q])
            for early, late in [
                *((r, r + 1) for r in range(15)),
                (0, 2),
                (0, 3),
                (0, 4),
            ]:
                features.extend(np.outer(shot[early], shot[late]).ravel())
            expected[point] += np.array([1, *features])

    actual = reduce_round(process, {0: iq, 7: np.zeros((3, 23, 1, 2))})

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(iq, original)
    np.testing.assert_allclose(
        reduce_round(process, {0: iq[:, :10]}) + reduce_round(process, {0: iq[:, 10:]}),
        actual,
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
            reducer=v2.thermal_chain_sums(prog, "readout"),
            rounds=2,
            progress=False,
        )
        expected = np.zeros((2, 108))
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
        reduce_round(process, {0: np.ones(shape)})


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


def test_thermal_rejects_repeat_present_only_in_one_setting():
    builder = v2.TimelineBuilder(load_soccfg())
    builder.add_loop("point", 2)
    readout = block_readout()
    with (
        builder.branch(by="point", align_timeline=True) as cases,
        cases.case(0),
        builder.repeat("read", 16),
    ):
        builder.emit(readout)
        builder.timeline_inc(1.0)
    with pytest.raises(ValueError, match=r"ADC 0.*expected 16.*actual.*\[\]"):
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
    iq += np.arange(16)[None, None, :, None] * np.array([0.2, -0.3])
    axis = np.array([1.0, 2.0])
    baseline = thermal_chain_score(
        ThermalChainSums.from_processor(axis, reduce_round(process, {0: iq}))
    )
    shifted = thermal_chain_score(
        ThermalChainSums.from_processor(
            axis, reduce_round(process, {0: iq + [1.5, -2.5]})
        )
    )
    for original, offset in zip(baseline.settings, shifted.settings, strict=True):
        np.testing.assert_allclose(
            [
                offset.setting,
                offset.score,
                offset.score_error,
                offset.correlation,
                offset.amplification,
                offset.qnd,
                offset.qnd_error,
                offset.mist,
            ],
            [
                original.setting,
                original.score,
                original.score_error,
                original.correlation,
                original.amplification,
                original.qnd,
                original.qnd_error,
                original.mist,
            ],
            rtol=1e-9,
        )
    assert shifted.ranking == baseline.ranking


def test_thermal_saved_adjacent_pairs_reconstruct_weighted_matrix_contrast():
    process = v2.thermal_chain_sums(block_program(16), "sample")
    rng = np.random.default_rng(209)
    # Position-dependent means and signal strength detect pooled centring and
    # endpoint-only contrasts.
    latent = rng.normal(size=(2, 2500, 1, 2))
    scales = np.linspace(0.5, 1.8, 16)[None, None, :, None]
    iq = scales * latent + rng.normal(size=(2, 2500, 16, 2))
    iq += np.arange(16)[None, None, :, None] * [0.3, -0.4]
    saved = ThermalChainSums.from_processor(
        np.array([1.0, 2.0]), reduce_round(process, {0: iq})
    )
    result = thermal_chain_score(saved)
    for point, shots in enumerate(iq):
        centred = shots - shots.mean(axis=0)
        adjacent = []
        for r in range(15):
            cross = np.zeros((2, 2))
            for shot in centred:
                cross += np.outer(shot[r], shot[r + 1]) / len(shots)
            adjacent.append((cross + cross.T) / 2)
        main = np.zeros((2, 2))
        for lag in range(1, 5):
            cross = np.zeros((2, 2))
            for shot in centred:
                cross += np.outer(shot[0], shot[lag]) / len(shots)
            main += (cross + cross.T) / 2
        axis = np.linalg.eigh(main)[1][:, -1]
        variance = np.mean((centred[:, 0] @ axis) ** 2)
        contrast = np.zeros((2, 2))
        for r, pair in enumerate(adjacent, start=1):
            contrast += (r - 8) / 20 * pair
        expected = np.linalg.eigvalsh(contrast)[-1] / variance
        np.testing.assert_allclose(result.settings[point].mist, expected, rtol=1e-9)
        # The saved data retains each pair for later independent reanalysis.
        raw = saved.feature_sums[point] / saved.shot_count[point]
        means = raw[:32].reshape(16, 2)
        for r, pair in enumerate(adjacent):
            cross = raw[35 + 4 * r : 39 + 4 * r].reshape(2, 2) - np.outer(
                means[r], means[r + 1]
            )
            np.testing.assert_allclose((cross + cross.T) / 2, pair, rtol=1e-9)


def test_thermal_rejects_undeclared_16_reads():
    builder = v2.TimelineBuilder(load_soccfg())
    with builder.repeat("shot", 16):
        builder.emit(block_readout())
        builder.delay_auto()
    with pytest.raises(ValueError, match=r"expected.*16.*actual.*1"):
        v2.thermal_chain_sums(builder.build(), "readout")


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
