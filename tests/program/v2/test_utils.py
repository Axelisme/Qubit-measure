import numpy as np
import pytest
from qick.asm_v2 import QickParam, QickProgramV2
from zcu_tools.program.v2 import utils
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import (
    param2str,
    readout_freq_from_word,
    readout_freq_from_words,
    readout_freq_words,
    sweep2param,
)


def test_sweep2param():
    cfg = SweepCfg(start=1.0, stop=5.0, expts=5, step=1.0)
    param = sweep2param("test_sweep", cfg)
    assert isinstance(param, QickParam)
    assert param.start == 1.0
    assert param.maxval() == 5.0
    assert param.minval() == 1.0


def test_param2str():
    # regular float
    assert param2str(3.14159) == "3.142"

    # integer
    assert param2str(10) == "10.000"

    # Format a QickParam without a sweep.
    param_no_sweep = QickParam(start=2.5, spans={})
    assert param2str(param_no_sweep) == "2.500"

    # Format a swept QickParam by its start and stop values.
    cfg = SweepCfg(start=1.0, stop=5.0, expts=5, step=1.0)
    param_sweep = sweep2param("test_sweep", cfg)
    assert param2str(param_sweep) == "sweep(1.000, 5.000)"


@pytest.mark.parametrize("mixer_freq", [None, 5000.123])
def test_readout_freq_words_match_qick_manager_waveforms(
    mixer_freq: float | None,
) -> None:
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    if mixer_freq is not None:
        soccfg["gens"][0]["has_mixer"] = True
    freq_mhz = 6000.0
    nqz = 2

    prog = QickProgramV2(soccfg)
    prog.declare_gen(0, nqz=nqz, mixer_freq=mixer_freq, ro_ch=0)
    prog.declare_readout(0, length=100)
    prog.add_pulse(
        0,
        "probe",
        ro_ch=0,
        style="const",
        freq=freq_mhz,
        phase=0.0,
        gain=0.5,
        length=1.0,
    )
    prog.add_readoutconfig(0, "adc", freq=freq_mhz, gen_ch=0)

    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        mixer_freq=mixer_freq,
        nqz=nqz,
        freq=0.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    gen_words, ro_words = readout_freq_words(
        soccfg,
        [freq_mhz],
        gen_ch=0,
        ro_ch=0,
        mixer_freq=mixer_freq,
        nqz=nqz,
    )

    gen_oracle = int(prog.pulses["probe"].waveforms[0].freq.start) % 2**32
    ro_oracle = int(prog.pulses["adc"].waveforms[0].freq.start) % 2**32
    assert gen_words == [gen_oracle]
    assert ro_words == [ro_oracle]
    assert 0 <= gen_words[0] < 2**32
    assert 0 <= ro_words[0] < 2**32

    assert readout_freq_from_word(
        soccfg,
        gen_words[0],
        kind="gen",
        cfg=cfg,
        reference_freq_mhz=freq_mhz,
    ) == pytest.approx(freq_mhz, abs=1e-5)

    assert readout_freq_from_words(
        soccfg,
        gen_words[0],
        ro_words[0],
        cfg=cfg,
    ) == pytest.approx(freq_mhz, abs=1e-5)
    assert readout_freq_from_word(
        soccfg,
        ro_words[0],
        kind="ro",
        cfg=cfg,
        reference_freq_mhz=freq_mhz,
    ) == pytest.approx(freq_mhz, abs=1e-5)


def test_readout_freq_from_words_rejects_corrupt_readout_word() -> None:
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        mixer_freq=None,
        nqz=1,
        freq=0.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    gen_words, ro_words = readout_freq_words(
        soccfg,
        [6000.0],
        gen_ch=0,
        ro_ch=0,
        mixer_freq=None,
        nqz=1,
    )

    with pytest.raises(ValueError, match="frequency words are inconsistent"):
        readout_freq_from_words(
            soccfg,
            gen_words[0],
            ro_words[0] ^ 1,
            cfg=cfg,
        )


def test_generator_word_decode_preserves_uint32_high_bit() -> None:
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    freq_mhz = 7000.0
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        mixer_freq=None,
        nqz=1,
        freq=0.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    gen_words, _ro_words = readout_freq_words(
        soccfg,
        [freq_mhz],
        gen_ch=0,
        ro_ch=0,
        mixer_freq=None,
        nqz=1,
    )

    assert gen_words[0] >= 2**31
    assert readout_freq_from_word(
        soccfg,
        gen_words[0],
        kind="gen",
        cfg=cfg,
    ) == pytest.approx(freq_mhz, abs=1e-5)


def test_generator_word_decode_handles_negative_mixer_relative_dds() -> None:
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    soccfg["gens"][0]["has_mixer"] = True
    mixer_freq = 5000.123
    freq_mhz = 4900.0
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        mixer_freq=mixer_freq,
        nqz=1,
        freq=0.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    gen_words, ro_words = readout_freq_words(
        soccfg,
        [freq_mhz],
        gen_ch=0,
        ro_ch=0,
        mixer_freq=mixer_freq,
        nqz=1,
    )

    assert gen_words[0] >= 2**31
    assert readout_freq_from_words(
        soccfg,
        gen_words[0],
        ro_words[0],
        cfg=cfg,
    ) == pytest.approx(freq_mhz, abs=1e-5)


@pytest.mark.parametrize("mixer_freq", [None, 5000.123])
def test_array_word_decoding_matches_scalar_bit_exactly(mixer_freq):
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    soccfg["gens"][0]["has_mixer"] = mixer_freq is not None
    freqs = [100.123, 7000.0, 4900.0]
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        mixer_freq=mixer_freq,
        nqz=1,
        freq=0.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    gen, adc = readout_freq_words(
        soccfg, freqs, gen_ch=0, ro_ch=0, mixer_freq=mixer_freq, nqz=1
    )
    expected_gen = np.array(
        [
            readout_freq_from_word(
                soccfg,
                w,
                kind="gen",
                cfg=cfg,
            )
            for w in gen
        ]
    )
    decoded = utils.readout_freqs_from_words(
        soccfg,
        np.array(gen, dtype=np.int64),
        kind="gen",
        cfg=cfg,
        paired_ro_words=np.array(adc, dtype=np.int64),
    )
    np.testing.assert_array_equal(decoded, expected_gen)
    expected_adc = np.array(
        [
            readout_freq_from_word(
                soccfg,
                w,
                kind="ro",
                cfg=cfg,
                reference_freq_mhz=f,
            )
            for w, f in zip(adc, expected_gen, strict=True)
        ]
    )
    np.testing.assert_array_equal(
        utils.readout_freqs_from_words(
            soccfg,
            np.array(adc, dtype=np.int64),
            kind="ro",
            cfg=cfg,
            reference_freq_mhz=decoded,
        ),
        expected_adc,
    )


def test_array_paired_word_validation_rejects_one_bit_corruption():
    soccfg = make_mock_soccfg(n_gens=1, n_readouts=1)
    cfg = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        mixer_freq=None,
        nqz=1,
        freq=0.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.1),
        ro_length=0.1,
    )
    gen, adc = readout_freq_words(
        soccfg, [100, 7000], gen_ch=0, ro_ch=0, mixer_freq=None, nqz=1
    )
    adc[1] ^= 1
    decode = utils.readout_freqs_from_words
    with pytest.raises(ValueError, match="frequency words are inconsistent"):
        decode(
            soccfg,
            np.array(gen, dtype=np.int64),
            kind="gen",
            cfg=cfg,
            paired_ro_words=np.array(adc, dtype=np.int64),
        )
