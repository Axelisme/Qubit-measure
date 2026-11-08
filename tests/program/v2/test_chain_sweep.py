"""Per-dwell chain statistics and summed acquisition through public interfaces."""

from fractions import Fraction

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program import v2
from zcu_tools.program.acquisition import ShotSumProcessor
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseReadout

from tests.program._profiles import load_soccfg


def chain_readout() -> PulseReadout:
    return PulseReadout(
        "readout",
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


def chain_program(n: int, *, reps: int = 1, points: int = 1) -> v2.TimelineProgramV2:
    builder = v2.TimelineBuilder(load_soccfg(), reps=reps)
    builder.add_loop("point", points)
    readout = chain_readout()
    with builder.repeat("chain", n):
        builder.emit(readout, tag="sample")
        builder.delay_auto()
    prog = builder.build()
    prog.set_reads_per_shot(n)
    return prog


def processor(prog: v2.TimelineProgramV2, read: str = "sample") -> ShotSumProcessor:
    return v2.chain_sweep_sums(prog, read)


def reference_vector(stream: NDArray[np.float64]) -> NDArray[np.float64]:
    """Scalar rational reference: explicitly enumerate pairs inside one dwell."""
    pairs = [(Fraction(float(i)), Fraction(float(q))) for i, q in stream]
    n = len(pairs)
    mean_i = sum(i for i, _q in pairs) / n
    mean_q = sum(q for _i, q in pairs) / n
    moments = [
        sum(i * i for i, _q in pairs) / n,
        sum(i * q for i, q in pairs) / n,
        sum(q * q for _i, q in pairs) / n,
    ]
    lagged = []
    for lags in ((1,), range(2, 16)):
        terms = [Fraction(0), Fraction(0), Fraction(0)]
        for lag in lags:
            for t in range(n - lag):
                i, q = pairs[t]
                later_i, later_q = pairs[t + lag]
                terms[0] += i * later_i / (n - lag)
                terms[1] += (i * later_q + q * later_i) / (n - lag)
                terms[2] += q * later_q / (n - lag)
        lagged.extend(terms)
    return np.array([1, mean_i, mean_q, *moments, *lagged], dtype=np.float64)


@pytest.mark.parametrize("n", [16, 20])
def test_chain_statistics_match_within_dwell_reference_and_augmented_sums(n: int):
    prog = chain_program(n, reps=2, points=2)
    ramp = np.arange(n, dtype=np.float64)
    iq = np.stack(
        [
            np.stack([ramp, 2 * ramp + 1], axis=-1),
            np.stack([ramp[::-1] - 4, ramp % 3], axis=-1),
            np.stack([ramp % 5, -ramp], axis=-1),
            np.stack([ramp + 2, ramp[::-1] ** 2], axis=-1),
        ]
    ).reshape(2, 2, n, 2)
    original = iq.copy()
    expected = np.zeros((2, 12, 12), dtype=np.float64)
    for point in range(2):
        for rep in range(2):
            z = reference_vector(iq[point, rep])
            expected[point] += np.outer(z, z)
    if n == 16:
        # Worked ramp: I=t, Q=2t+1, t=0..15; lag 1 has 15 pairs.
        np.testing.assert_allclose(
            reference_vector(iq[0, 0])[:9],
            [1, 7.5, 16, 77.5, 162.5, 341, 224 / 3, 941 / 3, 989 / 3],
        )
    actual = processor(prog)({0: iq})
    assert actual.shape == (2, 12, 12)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    np.testing.assert_array_equal(iq, original)


@pytest.mark.parametrize("n", [1, 15])
def test_chain_requires_at_least_sixteen_reads_at_construction(n: int):
    with pytest.raises(ValueError, match=rf"expected.*16.*actual.*{n}"):
        processor(chain_program(n))


@pytest.mark.parametrize("declared", [1, 15, 17])
def test_declared_read_count_must_match_fixed_repeat(declared: int):
    prog = chain_program(16)
    prog.set_reads_per_shot(declared)
    with pytest.raises(ValueError, match=rf"ADC 0.*expected.*16.*actual.*{declared}"):
        processor(prog)
    assert prog.ro_chs[0]["trigs"] == declared


@pytest.mark.parametrize(
    "shape", [(1, 2, 15, 2), (1, 2, 17, 2), (1, 2, 16, 3), (1, 16, 2)]
)
def test_processor_rejects_changed_read_count_or_iq_shape(shape: tuple[int, ...]):
    process = processor(chain_program(16))
    with pytest.raises(ValueError, match=r"ADC 0.*expected.*16, 2.*actual"):
        process({0: np.ones(shape, dtype=np.float64)})


def test_missing_read_label_lists_expected_actual_and_available_labels():
    with pytest.raises(
        ValueError, match=r"label 'absent'.*expected.*one.*actual 0.*sample"
    ):
        processor(chain_program(16), "absent")


def test_ambiguous_untagged_readout_label_reports_two_emissions():
    builder = v2.TimelineBuilder(load_soccfg())
    readout = chain_readout()
    with builder.repeat("chain", 16):
        builder.emit(readout)
        builder.delay_auto()
        builder.emit(readout)
        builder.delay_auto()
    prog = builder.build()
    prog.set_reads_per_shot(32)
    with pytest.raises(ValueError, match=r"label 'readout'.*expected one.*actual 2"):
        processor(prog, "readout")


@pytest.mark.parametrize("reps_innermost", [False, True])
def test_repeat_chain_acquisition_adds_dwell_counts_across_points_reps_and_rounds(
    reps_innermost: bool,
):
    env = white_noise(load_soccfg(), std_adc=0, seed=7, routes=((0, 0),))
    try:
        builder = v2.TimelineBuilder(env.soccfg, reps=3, reps_innermost=reps_innermost)
        builder.add_loop("point", 2)
        with builder.repeat("chain", 16):
            builder.emit(chain_readout())
            builder.delay_auto()
        prog = builder.build()
        prog.set_reads_per_shot(16)
        sums = prog.acquire_summed(
            env.soc, processor=processor(prog, "readout"), rounds=2, progress=False
        )
        expected = np.zeros((2, 12, 12), dtype=np.float64)
        expected[:, 0, 0] = 6
        np.testing.assert_array_equal(sums, expected)
        assert prog.ro_chs[0]["trigs"] == 16
    finally:
        env.devices.close_all_devices()
