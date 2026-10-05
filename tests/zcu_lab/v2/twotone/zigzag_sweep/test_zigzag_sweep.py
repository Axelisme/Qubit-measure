"""Zig-zag scan results survive a save/load round trip."""

from pathlib import Path

import numpy as np
from zcu_tools.datafile import load_labber_data
from zcu_tools.experiment.records import RunRecord

from zcu_lab.v2.twotone.zigzag_sweep.core import (
    ZigZagScanCfg,
    ZigZagScanExp,
    ZigZagScanResult,
)


def _cfg() -> ZigZagScanCfg:
    return ZigZagScanCfg.model_validate(
        {
            "reps": 2,
            "rounds": 1,
            "n_times": 6,
            "modules": {
                "X90_pulse": {
                    "ch": 1,
                    "nqz": 1,
                    "gain": 0.25,
                    "freq": 4000.0,
                    "waveform": {"style": "const", "length": 0.05},
                },
                "readout": {
                    "type": "readout/pulse",
                    "pulse_cfg": {
                        "ch": 0,
                        "nqz": 1,
                        "gain": 0.2,
                        "freq": 6000.0,
                        "waveform": {"style": "const", "length": 1.0},
                    },
                    "ro_cfg": {
                        "type": "readout/direct",
                        "ro_ch": 0,
                        "gen_ch": 0,
                        "ro_length": 0.4,
                        "ro_freq": 6000.0,
                        "trig_offset": 0.1,
                    },
                },
            },
            "sweep": {
                "gain": {"start": 0.4, "stop": 0.6, "step": 0.2 / 30, "expts": 31}
            },
        }
    )


def test_roundtrip_keeps_times_by_values_signals(tmp_path: Path) -> None:
    times = np.arange(7, dtype=np.int64)
    values = np.linspace(0.4, 0.6, 31)
    signals = (times[:, None] + 1j * values[None, :]).astype(np.complex128)
    source = RunRecord(
        cfg=_cfg(),
        result=ZigZagScanResult(times=times, values=values, signals=signals),
    )
    path = tmp_path / "zigzag_scan.hdf5"

    ZigZagScanExp().save(source, path)
    loaded = ZigZagScanExp().load(path)

    assert loaded.cfg == source.cfg
    np.testing.assert_array_equal(loaded.result.times, times)
    np.testing.assert_allclose(loaded.result.values, values)
    np.testing.assert_allclose(loaded.result.signals, signals)
    disk = load_labber_data(str(path))
    assert [axis.name for axis in disk.axes] == ["Sweep value", "Times"]
