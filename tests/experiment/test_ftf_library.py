"""The YAML offline builder and real public acquisition share numeric storage."""

import numpy as np
import yaml
from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.resources.ftf import parse_ftf

from tests.resources._ftf_data import ftf_document


def _program(env):
    drive = PulseCfg(
        ch=2, nqz=1, freq=0.0, gain=0.037, waveform=ConstWaveformCfg(length=0.008)
    )
    read = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=6000.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.016),
        ro_length=0.02,
    )
    return ModularProgramV2(
        env.soccfg,
        ProgramV2Cfg(reps=2, rounds=2),
        modules=[drive.build("mw"), read.build("ro")],
    )


def test_yaml_gpu_builder_and_public_closed_acquire_hit_table(tmp_path):
    from zcu_tools.experiment.ftf import FTFBuildPlan, build_database, ftf
    from zcu_tools.simulate.evolution import EvolutionLibrary

    settings = parse_ftf(ftf_document())
    coupler = settings.model.coupler.model_copy(update={"q_cap": None})
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"coupler": coupler})}
    )
    document = settings.model_dump(mode="json", exclude_none=True)
    yaml_path = tmp_path / "ftf.yaml"
    with yaml_path.open("w") as stream:
        yaml.safe_dump(document, stream)
    path = tmp_path / "data"
    built = build_database(
        yaml_path,
        path,
        FTFBuildPlan(8.0, (1 + 0j,), -0.1, 0.1, 6),
        tmp_path / "compiler",
    )
    assert built.published is not None
    library = EvolutionLibrary(path)
    env = ftf(yaml_path, library=library)
    direct = ftf(yaml_path)
    try:
        actual = _program(env).acquire_decimated(env.soc, progress=False)[0]
        assert library.last_evaluation is not None
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
        assert first_library.last_evaluation is not None
        assert first_library.last_evaluation.origin == "direct-record"
    finally:
        env.devices.close_all_devices()
    second_library = EvolutionLibrary(path)
    second = ftf(settings, library=second_library)
    try:
        actual = _program(second).acquire_decimated(second.soc, progress=False)[0]
        assert second_library.last_evaluation is not None
        assert second_library.last_evaluation.origin == "direct-record"
        np.testing.assert_array_equal(actual, reference)
        from zcu_tools.device.fake import FakeDevice

        driver = second.devices.get_device("bias")
        assert isinstance(driver, FakeDevice)
        driver.set_value(0.4)
        _program(second).acquire_decimated(second.soc, progress=False)
        assert second.soc.binding.rounds[0].snapshot.values == (("bias", 0.4),)
        assert len(tuple((path / "direct").glob("*.npz"))) >= 2
    finally:
        second.devices.close_all_devices()


def test_offline_cli_main_reads_yaml_and_resumes_partial(tmp_path):
    from zcu_tools.experiment.ftf.database import main
    from zcu_tools.simulate.evolution import EvolutionLibrary

    yaml_path = tmp_path / "ftf.yaml"
    with yaml_path.open("w") as stream:
        yaml.safe_dump(ftf_document(), stream)
    args = [
        str(yaml_path),
        str(tmp_path / "data"),
        "--duration-ns",
        "8",
        "--ratios",
        "1",
        "--lower",
        "-.1",
        "--upper",
        ".1",
        "--degree",
        "6",
        "--compiler-directory",
        str(tmp_path / "compiler"),
    ]
    assert main(args + ["--node-limit", "2"]) == 0
    assert not EvolutionLibrary(tmp_path / "data").snapshot().tables
    assert main(args) == 0
    assert len(EvolutionLibrary(tmp_path / "data").snapshot().tables) == 1


def test_closed_acquire_round_flux_does_not_hit_an_old_dc_table(tmp_path):
    """First-green coverage of actual source changes against published U content."""
    from zcu_tools.device.fake import FakeDevice
    from zcu_tools.experiment.ftf import FTFBuildPlan, build_database, ftf
    from zcu_tools.simulate.evolution import EvolutionLibrary

    settings = parse_ftf(ftf_document())
    coupler = settings.model.coupler.model_copy(update={"q_cap": None})
    settings = settings.model_copy(
        update={"model": settings.model.model_copy(update={"coupler": coupler})}
    )
    yaml_path = tmp_path / "ftf.yaml"
    with yaml_path.open("w") as stream:
        yaml.safe_dump(settings.model_dump(mode="json", exclude_none=True), stream)
    path = tmp_path / "data"
    build_database(
        yaml_path,
        path,
        FTFBuildPlan(8.0, (1 + 0j,), -0.1, 0.1, 6),
        tmp_path / "compiler",
    )
    library = EvolutionLibrary(path)
    env = ftf(settings, library=library)
    seen = []
    try:
        driver = env.devices.get_device("bias")
        assert isinstance(driver, FakeDevice)

        def change_bias(*_args, **_kwargs):
            assert library.last_evaluation is not None
            seen.append(library.last_evaluation.origin)
            driver.set_value(0.4)

        _program(env).acquire_decimated(env.soc, progress=False, round_hook=change_bias)
        assert seen == ["table", "direct"]
        assert env.soc.binding.rounds[1].snapshot.values == (("bias", 0.4),)
    finally:
        env.devices.close_all_devices()


def test_source_retains_each_round_group_actual_numeric_content_identity(tmp_path):
    from dataclasses import replace
    from zcu_tools.experiment.ftf import FTFSource, ftf
    from zcu_tools.simulate.evolution import EvolutionLibrary

    settings = parse_ftf(ftf_document())
    env = ftf(settings)
    try:
        _program(env).acquire_decimated(env.soc, progress=False)
        original = env.soc.binding.rounds[0]
        timelines = original.acquire.timelines
        second = timelines[1]
        events = tuple(replace(event, phase_deg=90.) if event.gen_ch == 2 else event for event in second.events)
        acquire = replace(original.acquire, timelines=(timelines[0], replace(second, events=events)))
        request = replace(original, acquire=acquire)
        source = FTFSource(settings, env.devices, library=EvolutionLibrary(tmp_path))
        source.execute(request)
        assert len(source.evaluations) == 2
        first = source.evaluations
        assert {item.round_index for item in first} == {0}
        assert {item.group_index for item in first} == {0,1}
        assert all(item.receipt.origin == "direct" for item in first)
        assert first[0].receipt.content_identities != first[1].receipt.content_identities
        source.execute(replace(request, index=1))
        assert len(source.evaluations) == 4
        for saved, repeated in zip(first, source.evaluations[2:], strict=True):
            assert repeated.round_index == 1
            assert repeated.receipt.origin == "direct-record"
            assert repeated.receipt.content_identities == saved.receipt.content_identities
        source.execute(replace(request, acquire=replace(acquire, sequence=acquire.sequence+1)))
        assert len(source.evaluations) == 2
        assert all(item.acquire_sequence == acquire.sequence+1 for item in source.evaluations)
    finally:
        env.devices.close_all_devices()
