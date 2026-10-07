"""Workflow native writer wiring through exact-path save and typed load seams."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import RunMetadata, RunSnapshot
from zcu_tools.experiment.acquisition import AcquisitionEvidence
from zcu_tools.experiment.native_persistence import load_run_with_metadata
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows import Completed, Run

from zcu_lab.v2.twotone.freq.core import FreqExp
from zcu_lab.v2.twotone.rabi.len_rabi.core import LenRabiExp
from zcu_lab.v2.twotone.ro_optimize.freq_gain.core import FreqGainExp
from zcu_lab.v2.twotone.time_domain.t1.core import T1Exp
from zcu_lab.workflows.acquire import (
    run_len_rabi,
    run_qubit_freq,
    run_ro_landscape,
    run_t1,
)
from zcu_lab.workflows.saving import save_frequency, save_rabi, save_readout, save_t1

from ._sim_inputs import Inputs, make_snapshot


def assert_history(
    metadata: RunMetadata, evidence: AcquisitionEvidence, snapshot: RunSnapshot
) -> None:
    """Check captured facts, not values recaptured at saving or loading time."""
    assert metadata.run_id == evidence.run_id
    assert metadata.started_at == evidence.started_at.isoformat()
    assert metadata.finished_at == evidence.finished_at.isoformat()
    assert metadata.completion == evidence.completion
    assert metadata.snapshot == snapshot
    assert metadata.provenance == evidence.provenance
    assert metadata.labber_path is None


def test_frequency_native_save_load_keeps_cfg_coordinates_and_history(
    inputs: Inputs, tmp_path: Path
) -> None:
    cfg = inputs.freq
    before = cfg.model_dump(mode="python")
    result = run_qubit_freq(
        Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    )
    path, snapshot = tmp_path / "spectroscopy.stage", make_snapshot()
    save_frequency(Completed(cfg, result), path, snapshot=snapshot)
    record, metadata = load_run_with_metadata(path, spec=FreqExp().persistence_spec())
    assert record.cfg is not None and record.cfg.model_dump(mode="python") == before
    np.testing.assert_allclose(record.result.freqs, result.freqs, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(record.result.signals, result.signals)
    assert_history(metadata, result.evidence, snapshot)
    assert cfg.model_dump(mode="python") == before


def test_rabi_native_save_load_keeps_cfg_coordinates_and_history(
    inputs: Inputs, tmp_path: Path
) -> None:
    cfg = inputs.rabi
    result = run_len_rabi(Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg))
    path, snapshot = tmp_path / "rabi.stage", make_snapshot()
    save_rabi(Completed(cfg, result), path, snapshot=snapshot)
    record, metadata = load_run_with_metadata(
        path, spec=LenRabiExp().persistence_spec()
    )
    assert record.cfg is not None
    assert record.cfg.model_dump(mode="python") == cfg.model_dump(mode="python")
    np.testing.assert_allclose(
        record.result.lengths, result.lengths, rtol=1e-14, atol=0
    )
    np.testing.assert_array_equal(record.result.signals, result.signals)
    assert_history(metadata, result.evidence, snapshot)


def test_readout_native_save_load_keeps_axis_order_snr_and_history(
    inputs: Inputs, tmp_path: Path
) -> None:
    cfg = inputs.readout
    result = run_ro_landscape(
        Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg)
    )
    path, snapshot = tmp_path / "readout.stage", make_snapshot()
    save_readout(Completed(cfg, result), path, snapshot=snapshot)
    record, metadata = load_run_with_metadata(
        path, spec=FreqGainExp().persistence_spec()
    )
    assert record.cfg is not None
    assert record.cfg.model_dump(mode="python") == cfg.model_dump(mode="python")
    np.testing.assert_allclose(record.result.freqs, result.freqs, rtol=1e-14, atol=0)
    np.testing.assert_allclose(record.result.gains, result.gains, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(record.result.signals, result.signals)
    assert_history(metadata, result.evidence, snapshot)


def test_t1_native_save_load_keeps_cfg_coordinates_and_history(
    inputs: Inputs, tmp_path: Path
) -> None:
    cfg = inputs.t1
    result = run_t1(Run(cfg, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg))
    path, snapshot = tmp_path / "decay.stage", make_snapshot()
    save_t1(Completed(cfg, result), path, snapshot=snapshot)
    record, metadata = load_run_with_metadata(path, spec=T1Exp().persistence_spec())
    assert record.cfg is not None
    assert record.cfg.model_dump(mode="python") == cfg.model_dump(mode="python")
    np.testing.assert_allclose(record.result.times, result.times, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(record.result.signals, result.signals)
    assert_history(metadata, result.evidence, snapshot)


def test_t1_saver_collision_preserves_published_native_data(
    inputs: Inputs, tmp_path: Path
) -> None:
    result = run_t1(Run(inputs.t1, StopSignal(), soc=inputs.soc, soccfg=inputs.soccfg))
    path, snapshot = tmp_path / "decay.stage", make_snapshot()
    source = Completed(inputs.t1, result)
    save_t1(source, path, snapshot=snapshot)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_t1(source, path, snapshot=snapshot)
    assert path.read_bytes() == before
