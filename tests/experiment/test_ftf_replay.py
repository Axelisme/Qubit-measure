"""Author-opted persistent round replay through acquisition and detached inputs."""

from pathlib import Path

import numpy as np
import pytest
from zcu_tools.device.fake import FakeDevice
from zcu_tools.experiment.ftf import ftf
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf
from zcu_tools.simulate.evolution import EvolutionLibrary

from tests.resources._ftf_data import ftf_document


def test_reload_selected_round_without_prior_acquires_or_live_devices(
    tmp_path: Path,
) -> None:
    settings = parse_ftf(
        ftf_document()
        | {
            "receiver": {
                "ro_ch": 0,
                "gain": 10000.0,
                "noise_psd_adc2_ns": [[30.0, 0.0], [0.0, 30.0]],
            }
        }
    )
    archive = tmp_path / "private"
    library = EvolutionLibrary(tmp_path / "numeric")
    env = ftf(settings, library=library, private_replay=archive)
    try:
        driver = env.devices.get_device("bias")
        if not isinstance(driver, FakeDevice):
            raise TypeError("FTF source must be a fixed-unit memory driver")
        program = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=4, rounds=3),
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000.0,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.016),
                    ro_length=0.016,
                ).build("ro")
            ],
        )
        values = iter((0.4, 0.2, 0.7))

        def change_bias(*_args, **_kwargs) -> None:
            driver.set_value(next(values))

        program.acquire(env.soc, progress=False, round_hook=change_bias)
        captured = env.soc.binding.rounds
        expected = env.soc.binding.execute(captured[2]).channels[0].iq.copy()
        driver.set_value(0.8)
        library.clear_memory()
    finally:
        env.devices.close_all_devices()

    from zcu_tools.experiment.ftf import replay_round
    from zcu_tools.resources.simulation_replay import read_replay, replay_paths

    paths = replay_paths(archive)
    selected = next(path for path in paths if read_replay(path).round.index == 2)
    restored = read_replay(selected)
    assert restored.round.snapshot.values == (("bias", 0.2),)
    assert restored.acquire.request == captured[2].acquire
    np.testing.assert_array_equal(replay_round(selected).channels[0].iq, expected)
    other = next(path for path in paths if read_replay(path).round.index == 0)
    # Same A bias, but independent round stream. No output/noise archive is reused.
    assert not np.array_equal(replay_round(other).channels[0].iq, expected)

    from zcu_tools.resources.simulation_replay import save_replay

    incompatible = restored.acquire.model_copy(
        update={"versions": (("backend", "future"),)}
    )
    mismatch = save_replay(
        tmp_path / "incompatible",
        incompatible,
        restored.round.snapshot,
        restored.round.index,
        restored.round.numeric_manifest,
    )
    with pytest.raises(ValueError, match="version mismatch"):
        replay_round(mismatch)

    # Numeric-owner references must be present even if the solver could recompute.
    records = tuple((library.path / "direct").glob("*.npz"))
    assert records
    for record in records:
        record.unlink()
    with pytest.raises(FileNotFoundError):
        replay_round(selected)
