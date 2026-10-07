"""The shipped LenRabiNode reaches a valid swept-pulse compile without acquire."""

from typing import NoReturn

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.gui.app.autofluxdep.nodes.builder import RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Snapshot
from zcu_tools.gui.cfg import SweepValue
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules import (
    DirectReadoutCfg,
    Pulse,
    PulseCfg,
    PulseReadoutCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.context import ModuleLibrary

import zcu_lab.v2.autofluxdep.lenrabi.autofluxdep as lenrabi
from tests.gui.app.autofluxdep._helpers import make_run_context


def test_lenrabi_node_generated_length_sweep_compiles_before_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[ModularProgramV2] = []

    def stop_at_acquisition(
        prog: ModularProgramV2, *_args: object, **_kwargs: object
    ) -> NoReturn:
        compiled.append(prog)
        raise RuntimeError("compile-only test boundary; acquisition is not permitted")

    def skip_flux_setup(_cfg: ExpCfgModel, _env: RunEnv, _exp_name: str) -> None:
        """Flux setup is outside this offline compile seam."""

    monkeypatch.setattr(ModularProgramV2, "acquire", stop_at_acquisition)
    monkeypatch.setattr(lenrabi, "setup_flux_point", skip_flux_setup)
    soc, soccfg = make_mock_soc()
    builder = lenrabi.LenRabiBuilder()
    schema = builder.make_default_schema().with_overrides(
        {
            "qub_ch": 0,
            "qub_nqz": 1,
            "qub_gain": 0.5,
            "sweep_range": SweepValue(start=0.04, stop=1.2, expts=9),
            "sweep_range_mode": "fixed",
            "drive_gain_mode": "fixed",
            "relax_delay_mode": "fixed",
            "relax_delay": 1.0,
            "reps": 1,
            "rounds": 1,
            "acquire_retry": 0,
        }
    )
    readout = PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=1, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=1.0)
        ),
        ro_cfg=DirectReadoutCfg(ro_ch=0, gen_ch=1, ro_freq=100.0, ro_length=1.0),
    )
    result = builder.make_init_result(schema, np.array([1.0]))
    env = RunEnv(
        flux=1.0,
        flux_idx=0,
        schema=schema,
        context=make_run_context(soc=soc, soccfg=soccfg),
        device_snapshot={},
        ml=ModuleLibrary(),
        result=result,
    )
    snapshot = Snapshot(
        {"qubit_freq": 1000.0},
        modules={"opt_readout": readout.model_dump(mode="python")},
    )
    with pytest.raises(RuntimeError, match="compile-only test boundary"):
        builder.build_node(env).produce(snapshot)

    # A compile failure never reaches the isolated acquisition boundary.
    assert len(compiled) == 1
    prog = compiled[0]
    assert prog.binprog is not None
    pulse = prog.modules[0]
    assert isinstance(pulse, Pulse)
    actual = np.asarray(
        prog.get_pulse_param(pulse.pulse_id, "total_length", as_array=True)
    )
    blocking = pulse.total_length(prog)
    assert isinstance(blocking, QickParam)
    timestamps = np.asarray(blocking.to_array(prog.loop_dict))
    assert actual.shape == timestamps.shape == (9,)
    assert np.all(timestamps >= actual - 1e-12)
