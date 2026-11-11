"""Six-source routing control through builder and program acquisition."""

import numpy as np
import pytest
from zcu_tools.cfg_model import MuxReadoutTone
from zcu_tools.program.v2 import TimelineBuilder
from zcu_tools.program.v2.modules.readout import MuxReadoutCfg
from zcu_tools.simulate.evolution import RoutingHistory
from zcu_tools.simulators.six_fluxonium.config import parse_six_fluxonium
from zcu_tools.simulators.six_fluxonium.environment import runtime

from tests.program._profiles import profile_soccfg
from tests.simulators.six_fluxonium.config._data import six_document

pytestmark = pytest.mark.sim


def routing_settings():
    document = six_document()
    for qubit in document["qubits"].values():
        qubit["chip"]["temperature_k"] = 0.05
        qubit["numerics"]["truncation"] = {"native_cutoff": 12, "levels": 2}
        qubit["readout"]["adc"]["noise_psd_adc2_ns"] = [[0.01, 0], [0, 0.01]]
    return parse_six_fluxonium(document)


def read_program(env):
    readout = MuxReadoutCfg(
        gen_ch=0,
        nqz=1,
        mixer_freq=6900,
        length=0.064,
        ro_length=0.026,
        trig_offset=0.008,
        tones=[
            MuxReadoutTone(ro_ch=ch, freq=6800 + 125 * ch, gain=0.1) for ch in (0, 1)
        ],
    )
    builder = TimelineBuilder(env.soccfg, reps=8, initial_delay=0, relax_delay=0)
    with builder.repeat("reads", 2):
        builder.emit(readout.build("readout"))
        builder.timeline_inc(0.096)
    builder.delay_by_timeline()
    program = builder.build()
    program.set_reads_per_shot(2)
    return program


def test_builder_routes_every_cavity_span_and_clear_resolves_again(monkeypatch):
    preparations = []

    class ObservedSource(runtime.SingleFluxoniumSource):
        def prepare_readouts(self, validated, *, cancel=None):
            result = super().prepare_readouts(validated, cancel=cancel)
            preparations.append(result)
            return result

    monkeypatch.setattr(runtime, "SingleFluxoniumSource", ObservedSource)
    history = RoutingHistory()
    env = runtime.six_fluxonium(
        routing_settings(),
        soccfg=profile_soccfg("six_fluxonium"),
        history=history,
        families={"t4"},
    )
    try:
        program = read_program(env)
        program.acquire(env.soc, progress=False)
        first = tuple(preparations)
        assert len(first) == 2
        for prepared in first:
            assert set(prepared.paths.segments_by_kernel) == {"t4"}
            diagnostics = prepared.projection_diagnostics
            assert diagnostics is not None
            assert diagnostics.evolution_calls >= 2
        program.acquire(env.soc, progress=False)
        assert all(
            new is old for new, old in zip(preparations[-2:], first, strict=True)
        )
        history.clear()
        program.acquire(env.soc, progress=False)
        assert all(
            new is not old for new, old in zip(preparations[-2:], first, strict=True)
        )
        assert all(
            set(new.paths.segments_by_kernel) == {"t4"} for new in preparations[-2:]
        )
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


def test_builder_rejects_unknown_family_before_acquire():
    with pytest.raises(ValueError, match="unknown kernel families"):
        runtime.six_fluxonium(
            routing_settings(),
            soccfg=profile_soccfg("six_fluxonium"),
            families={"unknown"},
        )


def test_acquire_rejects_domain_without_an_allowed_family():
    env = runtime.six_fluxonium(
        routing_settings(),
        soccfg=profile_soccfg("six_fluxonium"),
        families={"dop853"},
    )
    try:
        with pytest.raises(RuntimeError, match=r"domain idle: points \[0\], segments"):
            read_program(env).acquire(env.soc, progress=False)
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("borrowed", [False, True])
def test_builder_keeps_six_owned_histories_or_borrows_one_without_clearing_on_close(
    monkeypatch,
    borrowed,
):
    sources = []
    source_type = runtime.SixFluxoniumSource

    def observed_source(
        settings, devices, *, clock=None, truth_log=None, history=None, families=None
    ):
        source = source_type(
            settings,
            devices,
            clock=clock,
            truth_log=truth_log,
            history=history,
            families=families,
        )
        sources.append(source)
        return source

    monkeypatch.setattr(runtime, "SixFluxoniumSource", observed_source)
    shared = RoutingHistory() if borrowed else None
    env = runtime.six_fluxonium(
        routing_settings(),
        soccfg=profile_soccfg("six_fluxonium"),
        history=shared,
    )
    try:
        histories = tuple(
            source.model_facts(source.snapshot()).source.history
            for source in sources[0].sources
        )
        if borrowed:
            assert all(history is shared for history in histories)
        else:
            assert len({id(history) for history in histories}) == 6
        generations = tuple(history.generation for history in histories)
        sources[0].close()
        assert tuple(history.generation for history in histories) == tuple(
            generation if borrowed else generation + 1 for generation in generations
        )
    finally:
        if env.close_backend is not None:
            env.close_backend()
        env.devices.close_all_devices()


@pytest.mark.parametrize("mode", ["integrated", "decimated"])
def test_history_clear_preserves_complete_same_seed_cavity_iq_sequence(mode):
    cleared_history, retained_history = RoutingHistory(), RoutingHistory()
    config = routing_settings()
    cleared = runtime.six_fluxonium(
        config,
        soccfg=profile_soccfg("six_fluxonium"),
        history=cleared_history,
    )
    retained = runtime.six_fluxonium(
        config,
        soccfg=profile_soccfg("six_fluxonium"),
        history=retained_history,
    )
    try:
        cleared_program, retained_program = (
            read_program(cleared),
            read_program(retained),
        )
        for index in range(4):
            if index == 2:
                cleared_history.clear()
            if mode == "integrated":
                actual = cleared_program.acquire(cleared.soc, progress=False)
                expected = retained_program.acquire(retained.soc, progress=False)
            else:
                actual = cleared_program.acquire_decimated(cleared.soc, progress=False)
                expected = retained_program.acquire_decimated(
                    retained.soc, progress=False
                )
            for actual_iq, expected_iq in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(actual_iq, expected_iq)
            actual_raw, expected_raw = (
                cleared_program.get_raw(),
                retained_program.get_raw(),
            )
            assert actual_raw is not None and expected_raw is not None
            for actual_iq, expected_iq in zip(actual_raw, expected_raw, strict=True):
                np.testing.assert_array_equal(actual_iq, expected_iq)
    finally:
        for env in (cleared, retained):
            if env.close_backend is not None:
                env.close_backend()
            env.devices.close_all_devices()
