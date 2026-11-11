"""Builder-owned routing policy and author solve records through program acquire."""

import pytest
from zcu_tools.program.v2.modules.delay import SoftDelay
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.simulate.evolution import RoutingHistory
from zcu_tools.simulators.ftf.environment import runtime

from tests.program._profiles import profile_soccfg
from tests.program._timeline import compile_modules
from tests.simulators.ftf.environment._templates import settings

pytestmark = pytest.mark.sim


def routing_settings():
    config = settings().with_telemetry(enabled=False)
    return config.model_copy(
        update={
            "numerics": config.numerics.model_copy(
                update={
                    "truncation": config.numerics.truncation.model_copy(
                        update={"retained_levels": 2}
                    ),
                }
            ),
        }
    )


def read_program(env):
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=6000,
        gain=0.01,
        waveform=ConstWaveformCfg(length=0.008),
        ro_length=0.004,
        trig_offset=0.002,
    )
    return compile_modules(
        env.soccfg,
        modules=[
            SoftDelay("prefix", 0.008),
            readout.build("herald"),
            SoftDelay("wait", 0.024),
            readout.build("terminal"),
        ],
        reps=8,
        initial_delay=0,
        relax_delay=0,
    )


def test_builder_routes_every_projective_span_and_clear_resolves_again(monkeypatch):
    sources = []
    source_type = runtime.FTFSource

    def observed_source(
        settings,
        devices,
        *,
        soccfg,
        engine=None,
        acquisition=None,
        readout=None,
        telemetry=None,
        history=None,
    ):
        source = source_type(
            settings,
            devices,
            soccfg=soccfg,
            engine=engine,
            acquisition=acquisition,
            readout=readout,
            telemetry=telemetry,
            history=history,
        )
        sources.append(source)
        return source

    monkeypatch.setattr(runtime, "FTFSource", observed_source)
    history = RoutingHistory()
    env = runtime.ftf(
        routing_settings(),
        soccfg=profile_soccfg("ftf"),
        time_scale=0,
        history=history,
        families={"t4"},
    )
    try:
        program = read_program(env)
        program.acquire(env.soc, progress=False)
        source = sources[0]
        first = source.evaluations[0]
        assert set(first.paths.segments_by_kernel) == {"t4"}
        assert first.paths.segments_by_kernel["t4"] >= 2
        program.acquire(env.soc, progress=False)
        reused = source.evaluations[0].paths
        assert reused.segments_by_kernel == first.paths.segments_by_kernel
        program.acquire(env.soc, progress=False)
        assert source.evaluations[0].paths is reused
        history.clear()
        program.acquire(env.soc, progress=False)
        assert source.evaluations[0].paths is not reused
        assert (
            source.evaluations[0].paths.segments_by_kernel
            == first.paths.segments_by_kernel
        )
    finally:
        for source in sources:
            source.close()
        env.devices.close_all_devices()


def test_builder_rejects_unknown_family_before_acquire():
    with pytest.raises(ValueError, match="unknown kernel families"):
        runtime.ftf(
            routing_settings(), soccfg=profile_soccfg("ftf"), families={"unknown"}
        )


def test_acquire_rejects_domain_without_an_allowed_family():
    env = runtime.ftf(
        routing_settings(),
        soccfg=profile_soccfg("ftf"),
        time_scale=0,
        families={"dop853"},
    )
    try:
        with pytest.raises(RuntimeError, match=r"domain idle: points \[0\], segments"):
            read_program(env).acquire(env.soc, progress=False)
    finally:
        env.devices.close_all_devices()
