"""Cycle moments and acquisition through the public program interfaces."""

from contextlib import nullcontext

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.program import v2
from zcu_tools.program.v2.modules import ConstWaveformCfg, PulseReadout

from tests.program._profiles import load_soccfg
from tests.program._shot_reduction import reduce_round


def cycle_readout(name: str = "readout") -> PulseReadout:
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


def cycle_program(
    *, repeated: bool, reps: int = 1, points: int = 1, tagged: bool = False
) -> v2.TimelineProgramV2:
    builder = v2.TimelineBuilder(load_soccfg(), reps=reps)
    builder.add_loop("point", points)
    readout = cycle_readout()
    with builder.repeat("cycle", 8) if repeated else nullcontext():
        for block in range(1 if repeated else 8):
            builder.delay(0.2)  # D is a free-evolution interval in this cycle.
            for read in range(2):
                tag = "sample" if tagged and block == read == 0 else None
                builder.emit(readout, tag=tag)
                builder.delay_auto()
    return builder.build()


def reference_vector(stream: NDArray[np.float64]) -> NDArray[np.float64]:
    """Enumerate the DRR word to compute a cycle's moments with scalar sums."""
    word = "DRR" * 8
    positions = [pos for pos, letter in enumerate(word) if letter == "R"]
    singles: list[list[list[float]]] = [[], [], []]
    for read, (i, q) in enumerate(stream):
        group = 0 if read == 0 else 1 if word[positions[read] - 1] == "D" else 2
        singles[group].append([i * i, i * q, q * q])
    pair_terms: list[list[list[float]]] = [[] for _ in range(6)]
    for left in range(16):
        for right in range(left + 1, 16):
            between = word[positions[left] + 1 : positions[right]]
            d = between.count("D")
            gap = between.count("R") + 1
            group = {
                (0, 1): 0,
                (1, 1): 1,
                (1, 2): 2,
                (2, 2): 3,
                (2, 4): 4,
                (3, 4): 5,
                (3, 2): 5,
            }[min(d, 3), 1 if gap == 1 else 2 if gap <= 3 else 4]
            i, q = stream[left]
            j, r = stream[right]
            pair_terms[group].append([i * j, i * r, q * j, q * r])
    assert [len(group) for group in pair_terms] == [8, 7, 21, 6, 18, 60]
    return np.concatenate(
        [
            [1],
            stream.ravel(),
            *(np.sum(group, axis=0) for group in singles),
            *(np.sum(group, axis=0) for group in pair_terms),
        ]
    )


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("tagged", [False, True])
def test_cycle_moments_and_products_match_direct_reference(
    repeated: bool, tagged: bool
):
    prog = cycle_program(repeated=repeated, reps=2, points=2, tagged=tagged)
    ramp = np.arange(16, dtype=np.float64)
    streams = np.stack(
        [
            np.stack([ramp, 2 * ramp + 1], axis=-1),
            np.stack([ramp[::-1] - 4, ramp % 3], axis=-1),
            np.stack([ramp % 5, -ramp], axis=-1),
            np.stack([ramp + 2, ramp[::-1] ** 2], axis=-1),
        ]
    ).reshape(2, 2, 16, 2)
    original = streams.copy()
    expected = np.zeros((2, 66, 66))
    for point in range(2):
        for rep in range(2):
            z = reference_vector(streams[point, rep])
            expected[point] += np.outer(z, z)
    # Worked ramp: read zero, seven later after-D reads, eight after-R reads.
    np.testing.assert_array_equal(
        reference_vector(streams[0, 0])[33:42],
        [0, 0, 1, 560, 1176, 2471, 680, 1424, 2984],
    )
    actual = reduce_round(
        v2.chain_cycle_sums(prog, "sample" if tagged else "readout"),
        {0: streams, 1: np.zeros((2, 2, 3, 2))},
    )
    assert actual.shape == (2, 66, 66)
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=1e-13)
    np.testing.assert_array_equal(streams, original)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_read_label_requires_one_owner_and_lists_available_labels(ambiguous: bool):
    if ambiguous:
        builder = v2.TimelineBuilder(load_soccfg())
        builder.emit(cycle_readout())
        builder.delay_auto()
        builder.emit(cycle_readout("other"), tag="readout")
        prog = builder.build()
        label, count = "readout", 2
    else:
        prog = cycle_program(repeated=False)
        label, count = "absent", 0
    with pytest.raises(
        ValueError,
        match=rf"label '{label}'.*expected one readout owner.*actual {count}.*"
        r"available labels.*readout",
    ):
        v2.chain_cycle_sums(prog, label)


@pytest.mark.parametrize(
    "reads,repeats", [(15, None), (17, None), (1, 16), (4, 4), (2, 7), (2, 9)]
)
def test_source_reads_require_unrolled_cycle_or_two_reads_repeated_eight_times(
    reads: int, repeats: int | None
):
    builder = v2.TimelineBuilder(load_soccfg())
    readout = cycle_readout()
    with builder.repeat("wrong", repeats) if repeats is not None else nullcontext():
        for _ in range(reads):
            builder.emit(readout)
            builder.delay_auto()
    prog = builder.build()
    with pytest.raises(
        ValueError, match=rf"ADC 0.*expected.*16 unrolled.*actual.*{reads}"
    ):
        v2.chain_cycle_sums(prog, "readout")


@pytest.mark.parametrize("extra", ["raw trigger", "other owner"])
def test_cycle_rejects_other_triggers_on_its_adc(extra: str):
    builder = v2.TimelineBuilder(load_soccfg())
    readout = cycle_readout()
    with builder.repeat("cycle", 8):
        for _ in range(2):
            builder.emit(readout)
            builder.delay_auto()
    if extra == "raw trigger":
        builder.trigger(ros=[0])
    else:
        builder.emit(cycle_readout("other"))
    prog = builder.build()
    with pytest.raises(
        ValueError, match="ADC 0.*expected 2.*source triggers.*actual 3"
    ):
        v2.chain_cycle_sums(prog, "readout")


@pytest.mark.parametrize(
    "shape", [(1, 2, 15, 2), (1, 2, 17, 2), (1, 2, 16, 3), (1, 16, 2)]
)
def test_cycle_processor_rejects_incoming_read_or_iq_shape(shape: tuple[int, ...]):
    process = v2.chain_cycle_sums(cycle_program(repeated=False), "readout")
    with pytest.raises(ValueError, match="ADC 0.*expected.*16, 2.*actual"):
        reduce_round(process, {0: np.ones(shape, dtype=np.float64)})


@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("reps_innermost", [False, True])
def test_cycle_acquisition_adds_counts_across_points_reps_and_rounds(
    repeated: bool, reps_innermost: bool
):
    env = white_noise(load_soccfg(), std_adc=0, seed=7, routes=((0, 0),))
    try:
        builder = v2.TimelineBuilder(env.soccfg, reps=3, reps_innermost=reps_innermost)
        builder.add_loop("point", 2)
        readout = cycle_readout()
        with builder.repeat("cycle", 8) if repeated else nullcontext():
            for _ in range(1 if repeated else 8):
                builder.delay(0.2)
                builder.emit(readout)
                builder.delay_auto()
                builder.emit(readout)
                builder.delay_auto()
        prog = builder.build()
        sums = prog.acquire_summed(
            env.soc,
            reducer=v2.chain_cycle_sums(prog, "readout"),
            rounds=2,
            progress=False,
        )
        expected = np.zeros((2, 66, 66))
        expected[:, 0, 0] = 6
        np.testing.assert_array_equal(sums, expected)
        assert prog.ro_chs[0]["trigs"] == 16
    finally:
        env.devices.close_all_devices()
