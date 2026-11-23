"""Native arbitrary waveforms built without accessing a waveform asset."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qick import QickConfig
from zcu_tools.cfg_model import module_spec_from_raw, waveform_spec_from_raw
from zcu_tools.core.waveform_format import (
    WaveformFile,
    WaveformFormatError,
    WaveformSamples,
    encode_npz,
    hash_samples,
)
from zcu_tools.program.v2 import (
    InMemoryArbWaveformCfg,
    Pulse,
    PulseCfg,
    TimelineBuilder,
    WaveformCfg,
    WaveformCfgFactory,
)
from zcu_tools.program.v2.modules.waveform import ArbWaveformCfg, FlatTopWaveformCfg

from tests.program.v2.wave_observations import executed_words


@pytest.fixture
def samples() -> WaveformSamples:
    return WaveformSamples(
        time=np.array([0.0, 0.027, 0.1]),
        idata=np.array([0.0, 0.5, 0.0]),
        qdata=np.array([0.0, -0.25, 0.0]),
    )


def test_in_memory_waveform_keeps_a_read_only_sample_snapshot(
    samples: WaveformSamples,
) -> None:
    waveform = InMemoryArbWaveformCfg(samples=samples)
    expected_hash = hash_samples(samples)
    assert samples.qdata is not None
    samples.time[-1] = 0.2
    samples.idata[1] = 0.4
    samples.qdata[1] = 0.3

    for cfg in (waveform, deepcopy(waveform)):
        assert cfg.length == 0.1
        assert cfg.sample_hash == expected_hash == hash_samples(cfg.samples)
        assert cfg.samples.qdata is not None
        for array in (cfg.samples.time, cfg.samples.idata, cfg.samples.qdata):
            assert not array.flags.writeable
    with pytest.raises(ValueError, match="no sweepable parameters"):
        waveform.set_param("length", 0.2)

    with pytest.raises(WaveformFormatError, match="magnitude"):
        InMemoryArbWaveformCfg(
            samples=WaveformSamples(samples.time, np.array([0.0, 1.5, 0.0]), None)
        )


@pytest.mark.parametrize("method", ["to_dict", "model_dump", "model_dump_json"])
def test_in_memory_waveform_cannot_be_serialized(
    samples: WaveformSamples, method: str
) -> None:
    waveform = InMemoryArbWaveformCfg(samples=samples)
    pulse = PulseCfg(ch=0, nqz=1, freq=80, gain=0.2, waveform=waveform)
    for cfg in (waveform, pulse):
        with pytest.raises(ValueError, match="only exists in memory"):
            getattr(cfg, method)()


def test_in_memory_waveform_cannot_be_parsed(
    samples: WaveformSamples,
) -> None:
    raw = {"style": "arb_memory", "samples": samples}
    with pytest.raises(ValueError, match="WaveformSamples"):
        InMemoryArbWaveformCfg.model_validate({"samples": samples})
    with pytest.raises(ValueError, match="WaveformSamples"):
        WaveformCfgFactory.from_raw(raw)
    with pytest.raises(ValueError, match="WaveformSamples"):
        InMemoryArbWaveformCfg.model_validate_json(
            '{"style":"arb_memory","samples":{}}'
        )

    raw_spec = {
        "style": "arb_memory",
        "samples": {"time": [0.0, 1.0], "idata": [0.0, 0.0]},
    }
    with pytest.raises(ValueError, match="union_tag_invalid"):
        waveform_spec_from_raw(raw_spec)
    with pytest.raises(ValueError, match="union_tag_invalid"):
        module_spec_from_raw(
            {
                "type": "pulse",
                "ch": 0,
                "nqz": 1,
                "freq": 80,
                "gain": 0.2,
                "waveform": raw_spec,
            }
        )


@pytest.mark.parametrize("keep_q", [False, True])
def test_compiled_in_memory_envelopes_match_file_and_share_only_equal_samples(
    mock_soccfg: QickConfig, tmp_path: Path, samples: WaveformSamples, keep_q: bool
) -> None:
    if not keep_q:
        samples = replace(samples, qdata=None)
    file_cfg = _file_arb(tmp_path, samples)
    memory_cfg = InMemoryArbWaveformCfg(samples=samples)
    same_cfg = InMemoryArbWaveformCfg(samples=deepcopy(samples))
    other_cfg = InMemoryArbWaveformCfg(
        samples=replace(samples, idata=samples.idata / 2)
    )

    def build_pulse(name: str, waveform: WaveformCfg) -> Pulse:
        return Pulse(name, PulseCfg(ch=0, nqz=1, freq=80, gain=0.2, waveform=waveform))

    file = build_pulse("file", file_cfg)
    memory = build_pulse("memory", memory_cfg)
    same = build_pulse("same", same_cfg)
    different_knobs = Pulse("different_knobs", same.cfg, gain=0.3, phase=12, freq=85)
    other = build_pulse("other", other_cfg)
    builder = TimelineBuilder(mock_soccfg)
    for pulse in (file, memory, same, different_knobs, other):
        builder.emit(pulse)
    prog = builder.build()

    assert memory.pulse_name(prog) == same.pulse_name(prog)
    assert len({p.pulse_name(prog) for p in (memory, different_knobs, other)}) == 3
    table = prog.envelopes[0]["envs"]
    assert isinstance(table, dict)
    assert len(table) == 3  # File arb and the two different memory envelopes.
    file_iq, memory_iq, other_iq = (env["data"] for env in table.values())
    assert file_iq.shape == memory_iq.shape == other_iq.shape
    assert file_iq.tobytes() == memory_iq.tobytes() != other_iq.tobytes()


def _file_arb(tmp_path: Path, samples: WaveformSamples) -> ArbWaveformCfg:
    path = tmp_path / "shape.npz"
    path.write_bytes(encode_npz(WaveformFile(samples)))
    return ArbWaveformCfg(data="shape", path=str(path))


def _flat_top_builder(
    soccfg: QickConfig,
    ramp: ArbWaveformCfg | InMemoryArbWaveformCfg,
    *lengths: float,
) -> TimelineBuilder:
    builder = TimelineBuilder(soccfg)
    for length in lengths:
        waveform = FlatTopWaveformCfg(length=length, raise_waveform=ramp)
        pulse = PulseCfg(ch=0, nqz=1, freq=80, phase=15, gain=0.2, waveform=waveform)
        builder.emit(Pulse("drive", pulse))
    return builder


def test_flat_top_memory_ramp_compiles_like_asset(
    mock_soccfg: QickConfig, tmp_path: Path, samples: WaveformSamples
) -> None:
    lengths = (0.25, 0.4)
    asset = _flat_top_builder(
        mock_soccfg, _file_arb(tmp_path, samples), *lengths
    ).build()
    memory = _flat_top_builder(
        mock_soccfg, InMemoryArbWaveformCfg(samples=samples), *lengths
    ).build()

    # Different plateau lengths share the ramp but must retain their own playback.
    asset_table = asset.envelopes[0]["envs"]
    memory_table = memory.envelopes[0]["envs"]
    assert isinstance(asset_table, dict) and isinstance(memory_table, dict)
    (asset_env,) = asset_table.values()
    (memory_env,) = memory_table.values()
    np.testing.assert_array_equal(memory_env["data"], asset_env["data"])
    assert executed_words(memory) == executed_words(asset)


def test_short_flat_top_memory_ramp_fails_at_build_like_asset(
    mock_soccfg: QickConfig, tmp_path: Path, samples: WaveformSamples
) -> None:
    length = mock_soccfg.cycles2us(4, gen_ch=0)
    short = replace(samples, time=np.array([0.0, length / 2, length]))
    asset = _flat_top_builder(mock_soccfg, _file_arb(tmp_path, short), 0.25)
    with pytest.raises(RuntimeError, match="shorter than the min of 3") as expected:
        asset.build()

    memory = _flat_top_builder(mock_soccfg, InMemoryArbWaveformCfg(samples=short), 0.25)
    with pytest.raises(RuntimeError) as actual:
        memory.build()
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("method", ["to_dict", "model_dump_json"])
def test_flat_top_memory_ramp_cannot_be_serialized(
    samples: WaveformSamples, method: str
) -> None:
    ramp = InMemoryArbWaveformCfg(samples=samples)
    plain = PulseCfg(ch=0, nqz=1, freq=80, gain=0.2, waveform=ramp)
    with pytest.raises(ValueError, match="only exists in memory") as expected:
        getattr(plain, method)()

    flat_top = PulseCfg(
        ch=0,
        nqz=1,
        freq=80,
        gain=0.2,
        waveform=FlatTopWaveformCfg(length=0.25, raise_waveform=ramp),
    )
    with pytest.raises(ValueError, match="only exists in memory") as actual:
        getattr(flat_top, method)()
    assert str(actual.value) == str(expected.value)
