"""The YAML offline builder and real public acquisition share numeric storage."""

import numpy as np
import pytest
from ruamel.yaml import YAML

from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def _program(env):
    drive = PulseCfg(ch=2, nqz=1, freq=0., gain=.037, waveform=ConstWaveformCfg(length=.008))
    read = PulseReadoutCfg(gen_ch=0, ro_ch=0, nqz=1, freq=6000., gain=.1,
                          waveform=ConstWaveformCfg(length=.016), ro_length=.02)
    return ModularProgramV2(env.soccfg, ProgramV2Cfg(reps=2, rounds=2), modules=[drive.build("mw"), read.build("ro")])


def test_yaml_gpu_builder_and_public_closed_acquire_hit_table(tmp_path):
    from zcu_tools.experiment.ftf import FTFBuildPlan, build_database, ftf
    from zcu_tools.simulate.evolution import EvolutionLibrary

    settings = parse_ftf(ftf_document())
    coupler = settings.model.coupler.model_copy(update={"q_cap": None})
    settings = settings.model_copy(update={"model": settings.model.model_copy(update={"coupler": coupler})})
    document = settings.model_dump(mode="json", exclude_none=True)
    yaml_path = tmp_path / "ftf.yaml"
    with yaml_path.open("w") as stream:
        YAML().dump(document, stream)
    path = tmp_path / "data"
    built = build_database(yaml_path, path, FTFBuildPlan(8., (1+0j,), -.1, .1, 6), tmp_path / "compiler")
    assert built.published is not None
    library = EvolutionLibrary(path)
    env = ftf(yaml_path, library=library)
    direct = ftf(yaml_path)
    try:
        actual = _program(env).acquire_decimated(env.soc, progress=False)[0]
        assert library.last_evaluation.origin == "table"
        assert built.published.identity in library.last_evaluation.content_identities
        expected = _program(direct).acquire_decimated(direct.soc, progress=False)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
    finally:
        env.devices.close_all_devices()
        direct.devices.close_all_devices()


def test_nonzero_bath_persistent_records_and_live_round_flux(tmp_path):
    from zcu_tools.experiment.ftf import ftf
    from zcu_tools.simulate.evolution import EvolutionLibrary

    settings = parse_ftf(ftf_document())
    path = tmp_path / "data"
    first_library = EvolutionLibrary(path)
    env = ftf(settings, library=first_library)
    try:
        reference = _program(env).acquire_decimated(env.soc, progress=False)[0]
        assert first_library.last_evaluation.origin == "direct-record"
    finally:
        env.devices.close_all_devices()
    second_library = EvolutionLibrary(path)
    second = ftf(settings, library=second_library)
    try:
        actual = _program(second).acquire_decimated(second.soc, progress=False)[0]
        assert second_library.last_evaluation.origin == "direct-record"
        np.testing.assert_array_equal(actual, reference)
        from zcu_tools.device.fake import FakeDevice
        driver = second.devices.get_device("bias")
        assert isinstance(driver, FakeDevice)
        driver.set_value(.4)
        _program(second).acquire_decimated(second.soc, progress=False)
        assert second.soc.binding.rounds[0].snapshot.values == (("bias", .4),)
        assert len(tuple((path / "direct").glob("*.npz"))) >= 2
    finally:
        second.devices.close_all_devices()


def test_offline_cli_main_reads_yaml_and_resumes_partial(tmp_path):
    from zcu_tools.experiment.ftf.database import main
    from zcu_tools.simulate.evolution import EvolutionLibrary

    yaml_path = tmp_path / "ftf.yaml"
    with yaml_path.open("w") as stream:
        YAML().dump(ftf_document(), stream)
    args = [str(yaml_path), str(tmp_path / "data"), "--duration-ns", "8", "--ratios", "1", "--lower", "-.1", "--upper", ".1", "--degree", "6", "--compiler-directory", str(tmp_path / "compiler")]
    assert main(args+["--node-limit", "2"]) == 0
    assert not EvolutionLibrary(tmp_path / "data").snapshot().tables
    assert main(args) == 0
    assert len(EvolutionLibrary(tmp_path / "data").snapshot().tables) == 1
