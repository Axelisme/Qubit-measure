"""Public acquisition classifies normalized IQ without depending on lab cores."""

import logging
from threading import Event

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.singleshot import classify_result
from zcu_tools.cfg_model import CircleClassifier, LinearClassifier
from zcu_tools.program.acquisition import (
    AcquisitionContractError,
    ImproveAcquireMixin,
    StoppedPartialAcquireError,
)
from zcu_tools.program.herald import HeraldSpec
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.program._acquisition_soc import AcquisitionSoc, make_acquisition_soc


def _herald_fraction(
    program: ImproveAcquireMixin, channel: int = 0
) -> NDArray[np.float64]:
    fractions = program.get_herald_pass_fraction()
    assert fractions is not None, "completed herald round must publish pass fractions"
    return fractions[channel]


@pytest.mark.parametrize("radius", [0.5, 1.0, 1.5, 2.0, 3.0])
@pytest.mark.parametrize("remove_offset", [False, True])
def test_acquire_population_matches_offline_classification(
    monkeypatch: pytest.MonkeyPatch, radius: float, remove_offset: bool
) -> None:
    signals = np.array([-1, 1, 0, 0.2, -0.2, 4, -1 + radius], dtype=np.complex128)
    expected = classify_result(signals, -1, 1, radius)
    assert np.all(sum(mask.astype(int) for mask in expected) == 1)
    assert expected[2][2]  # An exact tie belongs to neither population.
    soc, soccfg = make_acquisition_soc()
    soccfg["readouts"][0]["iq_offset"] = 0.25
    program = ModularProgramV2(
        soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_ch=0,
                ro_length=soccfg.cycles2us(100, ro_ch=0),
                freq=100.0,
            ).build("readout")
        ],
        sweep=[("sample", len(signals))],
        reps=1,
    )
    raw = np.rint(
        np.stack((signals.real, signals.imag), axis=-1) * program.ro_chs[0]["length"]
    ).astype(np.int64)
    delivered = False

    def poll_data(totaltime: float = 0.1, timeout: float | None = None):
        nonlocal delivered
        if delivered:
            return []
        delivered = True
        return [(len(signals), ([raw], {}))]

    monkeypatch.setattr(soc, "poll_data", poll_data)
    actual = program.acquire(
        soc,
        classify={
            0: CircleClassifier(g_center=-1 + 0j, e_center=1 + 0j, radius=radius)
        },
        remove_offset=remove_offset,
        progress=False,
    )
    assert actual is not None
    np.testing.assert_array_equal(actual[0][0], np.stack(expected[:2], axis=-1))
    assert actual[0].dtype == np.float64


def test_typed_circle_uses_raw_coordinates_without_offset() -> None:
    soc, soccfg = make_acquisition_soc()
    soccfg["readouts"][0]["iq_offset"] = 10.0
    cfg = PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=1.0),
        ro_ch=0,
        ro_length=1.0,
        freq=100.0,
    )
    program = ModularProgramV2(soccfg, modules=[cfg.build("readout")], reps=2)
    result = program.acquire(
        soc,
        classify={0: CircleClassifier(g_center=0j, e_center=1 + 0j, radius=0.5)},
        remove_offset=True,
        progress=False,
    )
    assert result is not None
    np.testing.assert_array_equal(result[0], [[1.0, 0.0]])


@pytest.mark.parametrize("angle", [np.pi / 2, -np.pi / 2])
def test_linear_rotates_in_ge_theta_direction(
    monkeypatch: pytest.MonkeyPatch, angle: float
) -> None:
    soc, soccfg = make_acquisition_soc()
    cfg = PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=1.0),
        ro_ch=0,
        ro_length=soccfg.cycles2us(100, ro_ch=0),
        freq=100.0,
    )
    program = ModularProgramV2(
        soccfg, modules=[cfg.build("readout")], reps=1, sweep=[("sample", 3)]
    )
    raw = np.array([[0, 100], [0, -100], [0, 0]], dtype=np.int64)
    monkeypatch.setattr(soc, "poll_data", lambda: [(3, ([raw], {}))])
    result = program.acquire(
        soc,
        classify={0: LinearClassifier(angle=angle, threshold=0.0)},
        progress=False,
    )
    expected = [[1, 0], [0, 1], [1, 0]] if angle > 0 else [[0, 1], [1, 0], [1, 0]]
    assert result is not None
    np.testing.assert_array_equal(result[0][0], expected)


def test_classification_matches_adc_numbers_in_declaration_order() -> None:
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=2)
    soc = AcquisitionSoc(soccfg.get_cfg())
    program = ModularProgramV2(
        soccfg,
        reps=2,
        modules=[
            PulseReadoutCfg(
                gen_ch=ch,
                nqz=1,
                gain=0.1,
                freq=100.0,
                ro_ch=ch,
                waveform=ConstWaveformCfg(length=1.0),
                ro_length=1.0,
            ).build(f"readout_{ch}")
            for ch in (1, 0)
        ],
    )
    result = program.acquire(
        soc,
        progress=False,
        classify={
            0: CircleClassifier(g_center=0j, e_center=1 + 0j, radius=0.5),
            1: LinearClassifier(angle=0.0, threshold=-0.5),
        },
    )
    assert result is not None
    np.testing.assert_array_equal(result, [[[0, 1]], [[1, 0]]])
    iq = program.acquire(soc, progress=False)
    assert iq is not None
    np.testing.assert_array_equal(iq, 0)


@pytest.mark.parametrize("channels", [[], [0, 1]])
def test_classification_requires_exact_declared_adc_keys(
    monkeypatch: pytest.MonkeyPatch, channels: list[int]
) -> None:
    soc, program = _make_chunk_acquisition()

    def unexpected_poll():
        pytest.fail("invalid classification reached acquisition transport")

    monkeypatch.setattr(soc, "poll_data", unexpected_poll)
    with pytest.raises(ValueError, match="ADC.*missing=.*unexpected="):
        program.acquire(
            soc,
            classify={
                ch: LinearClassifier(angle=0.0, threshold=0.0) for ch in channels
            },
            progress=False,
        )


def test_missing_circle_radius_is_rejected_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soc, program = _make_chunk_acquisition()

    def unexpected_poll():
        pytest.fail("incomplete calibration reached acquisition transport")

    monkeypatch.setattr(soc, "poll_data", unexpected_poll)
    with pytest.raises(ValueError, match="radius.*Post"):
        program.acquire(
            soc,
            classify={0: CircleClassifier(g_center=0j, e_center=1 + 0j)},
            progress=False,
        )


@pytest.mark.parametrize("remove_offset", [False, True])
def test_herald_weights_completed_rounds_by_passed_shots(
    monkeypatch: pytest.MonkeyPatch,
    remove_offset: bool,
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    program.soccfg["readouts"][0]["iq_offset"] = 0.25
    length = program.ro_chs[0]["length"]
    # Round one passes one shot (signal 2); round two passes two (4 and 6).
    rounds = iter(
        np.array(values, dtype=np.int64) * length
        for values in (
            [[-1, 0], [2, 1], [1, 0], [90, 9], [1, 0], [90, 9]],
            [[-1, 0], [4, 2], [-1, 0], [6, 3], [1, 0], [90, 9]],
        )
    )
    monkeypatch.setattr(soc, "poll_data", lambda: [(3, ([next(rounds)], {}))])
    result = program.acquire(
        soc,
        herald=[HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)],
        rounds=2,
        progress=False,
        remove_offset=remove_offset,
    )
    # All three accepted shots together: [(2+4+6)/3, (1+2+3)/3].
    np.testing.assert_allclose(
        result[0], [[3.75, 1.75]] if remove_offset else [[4.0, 2.0]]
    )
    fraction = _herald_fraction(program)
    np.testing.assert_array_equal(fraction, 0.5)
    # An unswept accessor still returns a detached, writable scalar array.
    fraction[...] = -1
    np.testing.assert_array_equal(_herald_fraction(program), 0.5)


@pytest.mark.parametrize(
    "classifier",
    [
        LinearClassifier(angle=0.0, threshold=0.0),
        CircleClassifier(g_center=-1 + 0j, e_center=1 + 0j, radius=0.5),
    ],
)
def test_herald_masks_shots_before_population_classification(
    monkeypatch: pytest.MonkeyPatch,
    classifier,
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    program.soccfg["readouts"][0]["iq_offset"] = 100.0
    length = program.ro_chs[0]["length"]
    rounds = iter(
        np.array(values, dtype=np.int64) * length
        for values in (
            [[-1, 0], [-1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
            [[-1, 0], [1, 0], [-1, 0], [1, 0], [1, 0], [-1, 0]],
        )
    )
    monkeypatch.setattr(soc, "poll_data", lambda: [(3, ([next(rounds)], {}))])
    result = program.acquire(
        soc,
        herald=[HeraldSpec(0, 0, classifier, 2)],
        classify={0: classifier},
        rounds=2,
        progress=False,
    )
    np.testing.assert_allclose(result[0], [[1 / 3, 2 / 3]])


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("trackers", "herald.*trackers"),
        ("decimated", "decimated.*herald"),
        ("duplicate", "ADC 0.*one herald"),
        ("unknown_adc", "ADC 1.*declared"),
        ("read_count", "ADC 0.*reads_per_shot.*3.*2"),
        ("final_read", "ADC 0.*read_index.*later read"),
        ("negative_read", "ADC 0.*read_index"),
    ],
)
def test_herald_rejects_incompatible_options_and_layout_before_transport(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    soc, program = _make_chunk_acquisition()
    classifier = LinearClassifier(angle=0.0, threshold=0.0)
    spec = HeraldSpec(0, 0, classifier, 2)

    def unexpected_poll():
        pytest.fail("invalid herald acquisition reached transport")

    monkeypatch.setattr(soc, "poll_data", unexpected_poll)
    specs = {
        "duplicate": [spec, spec],
        "unknown_adc": [HeraldSpec(1, 0, classifier, 2)],
        "read_count": [HeraldSpec(0, 0, classifier, 3)],
        "final_read": [HeraldSpec(0, 1, classifier, 2)],
        "negative_read": [HeraldSpec(0, -1, classifier, 2)],
    }
    if case == "decimated":
        with pytest.raises(ValueError, match=message):
            program.acquire_decimated(soc, herald=[spec], progress=False)
    elif case == "trackers":
        with pytest.raises(ValueError, match=message):
            program.acquire(soc, herald=[spec], trackers=[], progress=False)
    else:
        with pytest.raises(ValueError, match=message):
            program.acquire(soc, herald=specs[case], progress=False)


@pytest.mark.parametrize("typed_classify", [False, True])
@pytest.mark.parametrize("angle", [0.0, np.pi / 2])
def test_herald_preserves_qick_projection_and_typed_classify_precedence(
    monkeypatch: pytest.MonkeyPatch,
    typed_classify: bool,
    angle: float,
) -> None:
    soc, program = _make_chunk_acquisition(reps=3, sweep=[("point", 2)])
    program.soccfg["readouts"][0]["iq_offset"] = 2.0
    length = program.ro_chs[0]["length"]
    herald_values = ([[-1, 1], [1, 1], [1, 1]], [[-1, 1], [-1, 1], [1, 1]])
    signals = ([[-1, 8], [8, 8], [8, 8]], [[1, 8], [3, 8], [8, 8]])
    rounds = iter(
        np.stack(
            [np.stack([h, np.zeros_like(h)], axis=-1), np.stack([s, s], axis=-1)],
            axis=-2,
        )
        .astype(np.int64)
        .reshape(-1, 2)
        * length
        for h, s in zip(
            map(np.array, herald_values), map(np.array, signals), strict=True
        )
    )
    monkeypatch.setattr(soc, "poll_data", lambda: [(6, ([next(rounds)], {}))])
    result = program.acquire(
        soc,
        herald=[HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)],
        classify={0: LinearClassifier(angle=0.0, threshold=100.0)}
        if typed_classify
        else None,
        threshold=-1.0 if angle == 0 else [-1.0],
        angle=angle if angle == 0 else [angle],
        rounds=2,
        progress=False,
    )
    # Projection after offset has accepted values -3, -1 (equality), +1.
    expected = np.array(
        [[[1.0, 0.0] if typed_classify else [1 / 3, 0.0], [np.nan, np.nan]]]
    )
    np.testing.assert_allclose(result[0], expected, equal_nan=True)
    np.testing.assert_array_equal(_herald_fraction(program), [0.5, 0])


@pytest.mark.parametrize("stepped", [False, True])
def test_herald_preserves_grid_and_other_reads_with_zero_pass_points(
    monkeypatch: pytest.MonkeyPatch,
    stepped: bool,
) -> None:
    soc, program = _make_chunk_acquisition(
        reps=2, reads=3, sweep=[("outer", 2), ("inner", 2)]
    )
    assert program.get_herald_pass_fraction() is None
    length = program.ro_chs[0]["length"]
    herald_values = (
        [[[1, 1], [-1, -1]], [[1, 1], [1, -1]]],
        [[[-1, 1], [-1, -1]], [[1, 1], [-1, -1]]],
    )
    signal_values = (
        [[[90, 90], [4, 10]], [[90, 90], [90, 12]]],
        [[[2, 90], [6, 14]], [[90, 90], [8, 16]]],
    )
    rounds = []
    for herald, signal in zip(herald_values, signal_values, strict=True):
        signal = np.array(signal)
        h = np.array(herald)
        rounds.append(
            np.stack(
                [
                    np.stack([signal, -signal], axis=-1),
                    np.stack([h, np.zeros_like(h)], axis=-1),
                    np.stack([10 * signal, -10 * signal], axis=-1),
                ],
                axis=-2,
            ).astype(np.int64)
            * length
        )
    batches = iter(rounds)
    monkeypatch.setattr(
        soc, "poll_data", lambda: [(8, ([next(batches).reshape(-1, 2)], {}))]
    )
    hooks = []

    def hook(count, data, _stop):
        hooks.append((count, data[0].copy()))

    spec = HeraldSpec(0, 1, LinearClassifier(angle=0.0, threshold=0.0), 3)
    if stepped:
        program.acquire(
            soc,
            herald=[spec],
            rounds=2,
            step_rounds=True,
            progress=False,
            round_hook=hook,
        )
        assert program.get_herald_pass_fraction() is None
        assert program.finish_round() is True
        program.prepare_round()
        assert program.finish_round() is False
        result = program.finish_acquire()
    else:
        result = program.acquire(
            soc, herald=[spec], rounds=2, progress=False, round_hook=hook
        )
    expected = np.array(
        [
            [[[2, -2], [np.nan, np.nan]], [[6, -6], [13, -13]]],
            [[[20, -20], [np.nan, np.nan]], [[60, -60], [130, -130]]],
        ]
    )
    assert result is not None
    np.testing.assert_allclose(result[0], expected, equal_nan=True)
    fractions = _herald_fraction(program)
    np.testing.assert_array_equal(fractions, [[0.25, 0], [0.75, 1]])
    fractions[:] = -1
    np.testing.assert_array_equal(_herald_fraction(program), [[0.25, 0], [0.75, 1]])
    assert [count for count, _ in hooks] == [1, 2]
    np.testing.assert_allclose(hooks[-1][1], expected, equal_nan=True)
    # Ordinary acquisition restores all reads and clears the accessor.
    monkeypatch.setattr(
        soc, "poll_data", lambda: [(8, ([np.zeros((24, 2), dtype=np.int64)], {}))]
    )
    ordinary = program.acquire(soc, progress=False)
    assert ordinary[0].shape == (3, 2, 2, 2)
    assert program.get_herald_pass_fraction() is None


@pytest.mark.parametrize("completed_round", [False, True])
def test_herald_cancellation_keeps_only_completed_round_statistics(
    monkeypatch: pytest.MonkeyPatch,
    completed_round: bool,
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    stop = Event()
    length = program.ro_chs[0]["length"]
    full = (
        np.array([[-1, 0], [2, 0], [1, 0], [90, 0], [1, 0], [90, 0]], dtype=np.int64)
        * length
    )
    partial = np.array([[-1, 0], [99, 0]], dtype=np.int64) * length
    batches = iter(
        ([(3, ([full], {}))], [(1, ([partial], {}))])
        if completed_round
        else ([(1, ([partial], {}))],)
    )

    def poll_data():
        batch = next(batches, None)
        if batch is not None:
            return batch
        stop.set()
        return []

    monkeypatch.setattr(soc, "poll_data", poll_data)
    spec = HeraldSpec(0, 0, LinearClassifier(angle=0.0, threshold=0.0), 2)
    program.acquire(
        soc, herald=[spec], rounds=2, progress=False, step_rounds=True, cancel_flag=stop
    )
    if completed_round:
        assert program.finish_round() is True
        program.prepare_round()
    assert program.finish_round() is False
    if completed_round:
        result = program.finish_acquire()
        assert result is not None
        np.testing.assert_array_equal(result[0], [[2, 0]])
        np.testing.assert_allclose(_herald_fraction(program), 1 / 3)
    else:
        with pytest.raises(StoppedPartialAcquireError):
            program.finish_acquire()
        assert program.get_herald_pass_fraction() is None


@pytest.mark.parametrize("second_herald", [False, True])
def test_herald_uses_adc_numbers_and_never_masks_other_adcs(
    monkeypatch: pytest.MonkeyPatch,
    second_herald: bool,
) -> None:
    soccfg = make_mock_soccfg(n_gens=2, n_readouts=2)
    soc = AcquisitionSoc(soccfg.get_cfg())
    program = ModularProgramV2(
        soccfg,
        reps=2,
        modules=[
            PulseReadoutCfg(
                gen_ch=ch,
                ro_ch=ch,
                nqz=1,
                freq=100.0,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_length=1.0,
            ).build(f"readout_{ch}_{read}")
            for read in range(2)
            for ch in (1, 0)
        ],
    )
    raw = [
        np.array(values, dtype=np.int64) * program.ro_chs[ch]["length"]
        for ch, values in (
            (1, [[-1, 0], [10, 0], [1, 0], [20, 0]]),
            (0, [[1, 0], [30, 0], [-1, 0], [40, 0]]),
        )
    ]
    monkeypatch.setattr(soc, "poll_data", lambda: [(2, (raw, {}))])
    classifier = LinearClassifier(angle=0.0, threshold=0.0)
    specs = [HeraldSpec(1, 0, classifier, 2)]
    if second_herald:
        specs.append(HeraldSpec(0, 0, classifier, 2))
    result = program.acquire(soc, herald=specs, progress=False)
    np.testing.assert_array_equal(result[0], [[10, 0]])
    np.testing.assert_array_equal(
        result[1], [[40, 0]] if second_herald else [[0, 0], [35, 0]]
    )
    fractions = program.get_herald_pass_fraction()
    assert fractions is not None
    assert set(fractions) == ({1, 0} if second_herald else {1})
    for fraction in fractions.values():
        np.testing.assert_array_equal(fraction, 0.5)


def _make_chunk_acquisition(
    *,
    reps: int = 4,
    reads: int = 2,
    sweep: list[tuple[str, int]] | None = None,
):
    soc, soccfg = make_acquisition_soc()
    readouts = [
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            gain=0.1 * (index + 1),
            waveform=ConstWaveformCfg(length=1.0),
            ro_ch=0,
            ro_length=1.0,
            freq=100.0,
        ).build(f"readout_{index}")
        for index in range(reads)
    ]
    program = ModularProgramV2(
        soccfg,
        modules=readouts,
        reps=reps,
        sweep=None if sweep is None else [(name, count) for name, count in sweep],
    )
    return soc, program


@pytest.mark.parametrize("stepped", [False, True])
def test_acquire_preserves_chunk_offsets_and_read_order(
    monkeypatch: pytest.MonkeyPatch, stepped: bool
) -> None:
    soc, program = _make_chunk_acquisition()
    raw = np.array(
        [
            [10, -10],
            [20, -20],
            [30, -30],
            [40, -40],
            [50, -50],
            [60, -60],
            [70, -70],
            [80, -80],
        ],
        dtype=np.int64,
    )
    # Multiple chunks in one poll, followed by another poll at a nonzero offset.
    batches = iter(
        [
            [(1, ([raw[:2]], {})), (2, ([raw[2:6]], {}))],
            [(1, ([raw[6:]], {}))],
        ]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: next(batches))
    if stepped:
        program.acquire(soc, progress=False, step_rounds=True)
        assert program.finish_round() is False
        result = program.finish_acquire()
    else:
        result = program.acquire(soc, progress=False)

    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(captured[0], raw.reshape(4, 2, 2))
    assert result is not None
    np.testing.assert_allclose(
        result[0], np.array([[40, -40], [50, -50]]) / program.ro_chs[0]["length"]
    )


def test_acquire_logs_size_mismatch_but_keeps_numpy_broadcasting(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    batches = iter(
        [
            [(2, ([np.array([[11, -11]], dtype=np.int64)], {}))],
            [(1, ([np.array([[21, -21], [22, -22]], dtype=np.int64)], {}))],
        ]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: next(batches))
    with caplog.at_level(logging.ERROR, logger="qick.qick_asm"):
        result = program.acquire(soc, progress=False)
    assert caplog.messages == [
        "data size mismatch: new_points=2, nreads=2, data shape (1, 2)"
    ]
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(
        captured[0],
        [[[11, -11], [11, -11]], [[11, -11], [11, -11]], [[21, -21], [22, -22]]],
    )
    assert result is not None
    np.testing.assert_allclose(
        result[0], np.array([[43, -43], [44, -44]]) / (3 * program.ro_chs[0]["length"])
    )


@pytest.mark.parametrize("stepped", [False, True])
@pytest.mark.parametrize("rows", [5, 6])
def test_acquire_rejects_shot_overrun_before_writing_chunk(
    monkeypatch: pytest.MonkeyPatch, stepped: bool, rows: int
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    first = np.array([[11, -11], [12, -12]], dtype=np.int64)
    batches = iter(
        [
            [(1, ([first], {}))],
            [(3, ([np.full((rows, 2), 17, dtype=np.int64)], {}))],
        ]
    )
    monkeypatch.setattr(soc, "poll_data", lambda: next(batches))
    completed_rounds: list[int] = []

    def record_round(count: int, _data: object, _stop: object) -> None:
        completed_rounds.append(count)

    message = (
        r"expected_shots=3, received_shots=1, chunk_shots=3, "
        rf"chunk_rows=\[{rows}\], round_index=0"
    )
    if stepped:
        program.acquire(soc, step_rounds=True, progress=False, round_hook=record_round)
        with pytest.raises(AcquisitionContractError, match=message):
            program.finish_round()
    else:
        with pytest.raises(AcquisitionContractError, match=message):
            program.acquire(soc, progress=False, round_hook=record_round)
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(
        captured[0], [[[11, -11], [12, -12]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]]
    )
    assert program.get_rounds() == []
    assert completed_rounds == []


@pytest.mark.parametrize("completed_round", [False, True])
def test_cancel_discards_partial_chunks_and_retains_only_completed_rounds(
    monkeypatch: pytest.MonkeyPatch, completed_round: bool
) -> None:
    soc, program = _make_chunk_acquisition(reps=3)
    stop = Event()
    full = np.array([[10, -10], [20, -20]] * 3, dtype=np.int64)
    batches = iter(
        [
            [(3, ([full], {}))],
            [(1, ([np.array([[91, -91], [92, -92]], dtype=np.int64)], {}))],
        ]
        if completed_round
        else [
            [(1, ([np.array([[91, -91], [92, -92]], dtype=np.int64)], {}))],
        ]
    )

    def poll_data():
        batch = next(batches, None)
        if batch is not None:
            return batch
        # The first partial chunk has been stored before this poll requests stop.
        stop.set()
        return [(2, ([np.full((4, 2), 99, dtype=np.int64)], {}))]

    monkeypatch.setattr(soc, "poll_data", poll_data)
    program.acquire(soc, rounds=2, progress=False, step_rounds=True, cancel_flag=stop)
    if completed_round:
        assert program.finish_round() is True
        program.prepare_round()
    assert program.finish_round() is False
    captured = program.get_raw()
    assert captured is not None
    np.testing.assert_array_equal(captured[0], np.zeros((3, 2, 2), dtype=np.int64))
    if completed_round:
        result = program.finish_acquire()
        assert result is not None
        np.testing.assert_allclose(
            result[0], np.array([[10, -10], [20, -20]]) / program.ro_chs[0]["length"]
        )
    else:
        with pytest.raises(
            StoppedPartialAcquireError, match="before the first round completed"
        ):
            program.finish_acquire()
