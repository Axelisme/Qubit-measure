from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import zcu_tools.experiment.v2.runner.schedule as schedule_module
import zcu_tools.experiment.v2.singleshot.len_rabi as len_rabi_module
from numpy.typing import NDArray
from zcu_tools.experiment.v2.singleshot.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiModuleCfg,
    LenRabiSweepCfg,
    classify_len_rabi_iq,
)
from zcu_tools.program.base import StoppedPartialAcquireError
from zcu_tools.program.v2 import DirectReadoutCfg, PulseCfg, SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import ConstWaveformCfg


def _cfg() -> LenRabiCfg:
    return LenRabiCfg(
        reps=2,
        rounds=7,
        shots=4,
        modules=LenRabiModuleCfg(
            reset=None,
            qub_pulse=PulseCfg(
                type="pulse",
                waveform=ConstWaveformCfg(style="const", length=1.0),
                ch=0,
                nqz=1,
                freq=3000.0,
                gain=0.2,
            ),
            readout=DirectReadoutCfg(
                type="readout/direct",
                ro_ch=0,
                ro_length=1.0,
                ro_freq=6000.0,
                gen_ch=0,
            ),
        ),
        sweep=LenRabiSweepCfg(
            length=SweepCfg(start=10.0, stop=30.0, step=10.0, expts=3),
        ),
    )


class _Viewer:
    def __init__(self) -> None:
        self.updates: list[NDArray[np.float64]] = []

    def __enter__(self) -> _Viewer:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_ax(self) -> Any:
        return self

    def set_ylim(self, *_args: float) -> None:
        return None

    def update(
        self,
        _lengths: NDArray[np.float64],
        populations: NDArray[np.float64],
    ) -> None:
        self.updates.append(populations.copy())


def test_len_rabi_run_preserves_raw_iq_and_forces_singleshot_cfg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    desired = np.array(
        [
            [-1.0 + 0.0j, -0.8 + 0.1j, 0.9 + 0.0j, 1.1 - 0.1j],
            [-1.1 + 0.0j, -0.9 + 0.1j, -0.7 + 0.0j, 0.8 + 0.0j],
            [0.7 + 0.0j, 0.9 + 0.1j, 1.0 + 0.0j, 1.2 - 0.1j],
        ],
        dtype=np.complex128,
    )
    acquire_kwargs: dict[str, object] = {}

    class RawProgram(ModularProgramV2):
        def acquire(self, _soc: object, **kwargs: object) -> list[NDArray[np.float64]]:
            acquire_kwargs.update(kwargs)
            return []

        def get_raw(self) -> list[NDArray[np.int64]]:
            length = float(next(iter(self.ro_chs.values()))["length"])
            qick_order = desired.T
            raw = np.zeros((*qick_order.shape, 1, 2), dtype=np.int64)
            raw[..., 0, 0] = np.rint(qick_order.real * length).astype(np.int64)
            raw[..., 0, 1] = np.rint(qick_order.imag * length).astype(np.int64)
            return [raw]

    viewer = _Viewer()
    monkeypatch.setattr(schedule_module, "ModularProgramV2", RawProgram)
    monkeypatch.setattr(len_rabi_module, "LivePlot1D", lambda *a, **k: viewer)
    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1, sim=None)

    with pytest.warns(UserWarning) as warnings_record:
        result = LenRabiExp().run(
            soc,
            soccfg,
            _cfg(),
            g_center=-1.0 + 0.0j,
            e_center=1.0 + 0.0j,
            radius=0.4,
        )

    assert len(warnings_record) == 2
    assert result.signals.dtype == np.complex128
    np.testing.assert_allclose(result.signals, desired, atol=0.011)
    assert result.signals.shape == (3, 4)
    np.testing.assert_array_equal(result.shot_indices, np.arange(4))
    assert result.cfg_snapshot is not None
    assert result.cfg_snapshot.rounds == 1
    assert result.cfg_snapshot.reps == result.cfg_snapshot.shots == 4
    assert "cancel_flag" in acquire_kwargs
    expected_populations = classify_len_rabi_iq(desired, -1.0 + 0.0j, 1.0 + 0.0j, 0.4)
    np.testing.assert_allclose(viewer.updates[-1][:2].T, expected_populations)


def test_len_rabi_run_rejects_non_qick_raw_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WrongLayoutProgram(ModularProgramV2):
        def acquire(self, _soc: object, **_kwargs: object) -> list[NDArray[np.float64]]:
            return []

        def get_raw(self) -> list[NDArray[np.int64]]:
            # Incorrectly sweep-first instead of QICK's reps-first layout.
            return [np.zeros((3, 4, 1, 2), dtype=np.int64)]

    monkeypatch.setattr(schedule_module, "ModularProgramV2", WrongLayoutProgram)
    monkeypatch.setattr(len_rabi_module, "LivePlot1D", lambda *a, **k: _Viewer())
    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1, sim=None)

    with pytest.warns(UserWarning):
        with pytest.raises(
            ValueError,
            match=r"expected \(3, 4\), got \(4, 3\)",
        ):
            LenRabiExp().run(
                soc,
                soccfg,
                _cfg(),
                g_center=-1.0 + 0.0j,
                e_center=1.0 + 0.0j,
                radius=0.4,
            )


def test_len_rabi_stopped_first_round_returns_nan_partial_raw_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StoppedProgram(ModularProgramV2):
        def acquire(self, _soc: object, **_kwargs: object) -> list[NDArray[np.float64]]:
            raise StoppedPartialAcquireError("stopped")

    monkeypatch.setattr(schedule_module, "ModularProgramV2", StoppedProgram)
    monkeypatch.setattr(len_rabi_module, "LivePlot1D", lambda *a, **k: _Viewer())
    soc, soccfg = make_mock_soc(n_gens=1, n_readouts=1, sim=None)

    with pytest.warns(UserWarning):
        result = LenRabiExp().run(
            soc,
            soccfg,
            _cfg(),
            g_center=-1.0 + 0.0j,
            e_center=1.0 + 0.0j,
            radius=0.4,
        )

    assert result.signals.shape == (3, 4)
    assert np.all(np.isnan(result.signals))
