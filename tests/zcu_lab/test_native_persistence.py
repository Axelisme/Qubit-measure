"""Native persistence seam with one shipped representative of each spec type.

Synthetic records exercise mapping only; no experiment run or instrument is used.
"""

from pathlib import Path

import numpy as np
from zcu_tools.experiment import RunRecord, load_run, save_run

from tests._native_support import native_metadata
from zcu_lab.v2.jpa.auto_optimize.core import (
    JPA_AUTO_GROUPED_AXES_SPEC,
    AutoOptimizeExp,
    JPAOptCfg,
    JPAOptimizeResult,
)
from zcu_lab.v2.onetone.freq.core import FreqCfg, FreqExp, FreqResult


def readout_cfg() -> dict[str, object]:
    """Return a synthetic pulse readout accepted by the shipped cfg models."""
    return {
        "type": "readout/pulse",
        "pulse_cfg": {
            "waveform": {"style": "const", "length": 1.0},
            "ch": 1,
            "nqz": 2,
            "freq": 6100.0,
            "gain": 0.2,
        },
        "ro_cfg": {"ro_ch": 2, "ro_freq": 6100.0, "ro_length": 1.0, "trig_offset": 0.5},
    }


def test_registered_single_spec_native_round_trip(tmp_path: Path) -> None:
    cfg = FreqCfg.model_validate(
        {
            "modules": {"readout": readout_cfg()},
            "sweep": {
                "freq": {"start": 4000.0, "stop": 5000.0, "expts": 2, "step": 1000.0}
            },
        }
    )
    record = RunRecord(
        cfg=cfg,
        result=FreqResult(np.array([4000.0, 5000.0]), np.array([1 + 2j, 3 - 4j])),
    )
    path = tmp_path / "single.h5"
    spec = FreqExp.AXES_SPEC
    assert spec is not None
    save_run(record, path, spec=spec, metadata=native_metadata(spec.tag))
    loaded, snapshot = load_run(path, spec=spec)
    assert loaded.cfg == cfg
    assert snapshot == native_metadata(spec.tag).snapshot
    np.testing.assert_array_equal(loaded.result.freqs, record.result.freqs)
    np.testing.assert_array_equal(loaded.result.signals, record.result.signals)


def test_registered_grouped_spec_native_round_trip(tmp_path: Path) -> None:
    sweep = {"start": 0.0, "stop": 1.0, "expts": 2, "step": 1.0}
    cfg = JPAOptCfg.model_validate(
        {
            "modules": {
                "pi_pulse": {
                    "type": "pulse",
                    "waveform": {"style": "const", "length": 0.05},
                    "ch": 3,
                    "nqz": 2,
                    "freq": 5000.0,
                    "gain": 0.3,
                },
                "readout": readout_cfg(),
            },
            "dev": {},
            "sweep": {"jpa_flux": sweep, "jpa_freq": sweep, "jpa_power": sweep},
            "num_points": 4,
        }
    )
    record = RunRecord(
        cfg=cfg,
        result=JPAOptimizeResult(
            params=np.array([[0.1, 6000.0, -10.0], [0.2, 6100.0, -9.0]]),
            phases=np.array([0, 1], dtype=np.int32),
            signals=np.array([1.5, 2.5]),
        ),
    )
    spec = JPA_AUTO_GROUPED_AXES_SPEC
    path = tmp_path / "grouped.h5"
    experiment = AutoOptimizeExp()
    experiment.save_run(record, path, metadata=native_metadata(spec.tag))
    loaded, snapshot = experiment.load_run(path)
    assert loaded.cfg == cfg
    assert snapshot == native_metadata(spec.tag).snapshot
    np.testing.assert_array_equal(loaded.result.params, record.result.params)
    np.testing.assert_array_equal(loaded.result.phases, record.result.phases)
    np.testing.assert_array_equal(loaded.result.signals, record.result.signals)
