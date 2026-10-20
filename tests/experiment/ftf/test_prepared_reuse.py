"""Source-owned noiseless preparations, fresh per-acquire streams and bounded reuse."""

from contextlib import ExitStack
from dataclasses import replace

import pytest
from zcu_tools.core.simulation import RoundRequest
from zcu_tools.device import DeviceManager
from zcu_tools.experiment.ftf import FTFSource, replay
from zcu_tools.program.v2.simulation import IntegratedRound

from tests.experiment._ftf_templates import (
    expect_statistics,
    request,
    settings,
    snapshot,
    template,
)

pytestmark = pytest.mark.sim


def draw(source, acquire=None, snap=None):
    return source.execute(
        RoundRequest(
            request() if acquire is None else acquire,
            snapshot() if snap is None else snap,
            0,
        )
    )


@pytest.fixture
def delayed_readout_template():
    """Leave room for shaped controls and retain each ADC's readout offset."""
    base = template(tones=1)
    readout = base.events[-1]
    shift = 0.140 - readout.start_us
    return replace(
        base,
        events=(*base.events[:-1], replace(readout, start_us=readout.start_us + shift)),
        readouts=tuple(
            replace(window, trigger_us=window.trigger_us + shift[:, None])
            for window in base.readouts
        ),
    )


def test_preparation_hits_preserve_source_usage_for_later_acquires(
    delayed_readout_template,
):
    import numpy as np
    from zcu_tools.core.simulation import PulseEnvelope

    base = delayed_readout_template
    drive, readout = base.events
    times = np.linspace(0.0, 0.128, 129)
    waveform = PulseEnvelope(
        tuple(times),
        tuple(np.exp(-0.5 * ((times - 0.064) / 0.032) ** 2).astype(complex)),
    )
    drive = replace(
        drive,
        start_us=np.zeros(2),
        duration_us=np.full(2, 0.128),
        gain=np.full(2, -0.002),
        envelopes=(waveform,),
    )
    readout = replace(readout, start_us=np.full(2, 0.140))
    acquire = request(templates=(replace(base, events=(drive, readout)),))
    histories, routes = [], []
    for clear in (False, True):
        source = FTFSource(settings(), DeviceManager())
        try:
            draw(source, acquire)
            numeric = source.model_facts(snapshot()).source
            first = dict(getattr(numeric, "gate_usage", {}))
            assert first, "shaped acquire must record its complete gate inventory"
            if clear:
                source.preparation_cache.clear()
            draw(source, replace(acquire, sequence=8))
            assert source.preparation_cache.hits == (0 if clear else 1)
            histories.append(dict(numeric.gate_usage))
            assert histories[-1] == first
            later_drive = replace(drive, phase_deg=drive.phase_deg + 17.0)
            later = replace(base, events=(later_drive, readout))
            draw(source, replace(acquire, templates=(later,), sequence=9))
            routes.append(source.evaluations[0].paths.segments_by_kernel)
            assert numeric.gate_usage == {
                key: 2 * count for key, count in first.items()
            }
            source.close()
            assert not numeric.gate_usage
        finally:
            source.close()
    assert histories[0] == histories[1]
    assert routes[0] == routes[1]


@pytest.mark.parametrize("rebuild", [False, True])
def test_cached_small_acquire_reselects_route_after_source_usage_grows(
    rebuild, delayed_readout_template
):
    import numpy as np
    from zcu_tools.core.simulation import PulseEnvelope

    base = delayed_readout_template
    drive, readout = base.events
    times = np.linspace(0.0, 0.128, 129)
    waveform = PulseEnvelope(
        tuple(times),
        tuple(np.exp(-0.5 * ((times - 0.064) / 0.032) ** 2).astype(complex)),
    )
    shaped = replace(
        base,
        events=(
            replace(
                drive,
                start_us=np.zeros(2),
                duration_us=np.full(2, 0.128),
                gain=np.full(2, -0.002),
                envelopes=(waveform,),
            ),
            replace(readout, start_us=np.full(2, 0.140)),
        ),
    )
    small = request(templates=(shaped,))
    large = replace(
        small,
        sequence=8,
        point_dims=(160,),
        loop_dims=(3, 160),
        templates=tuple(
            replace(shaped, points=shaped.points + 2 * i) for i in range(80)
        ),
    )
    routes = []
    for clear in (False, True):
        source = FTFSource(settings(), DeviceManager())
        try:
            draw(source, small)
            assert (
                source.evaluations[0].paths.segments_by_kernel.get("shaped_sambe") == 2
            )
            if rebuild:
                source.preparation_cache.clear()
                draw(source, replace(small, sequence=8))
            draw(source, large)
            if clear:
                source.preparation_cache.clear()
            draw(source, replace(small, sequence=9))
            routes.append(source.evaluations[0].paths.segments_by_kernel)
        finally:
            source.close()
    assert routes[0] == routes[1]
    assert routes[0].get("single_tone_gate") == 2


def test_history_sensitive_hit_reuses_policy_facts_without_prepare_or_put(
    delayed_readout_template,
):
    import numpy as np
    from zcu_tools.core.simulation import PulseEnvelope
    from zcu_tools.experiment.ftf.preparation import PreparationCache

    class CountingPreparations[T](PreparationCache[T]):
        def __init__(self):
            super().__init__()
            self.puts = 0

        def put(self, key, value):
            self.puts += 1
            super().put(key, value)

    source = FTFSource(settings(), DeviceManager())
    cache = CountingPreparations()
    source.preparation_cache = cache
    base = delayed_readout_template
    drive, readout = base.events
    times = np.linspace(0.0, 0.128, 129)
    waveform = PulseEnvelope(
        tuple(times),
        tuple(np.exp(-0.5 * ((times - 0.064) / 0.032) ** 2).astype(complex)),
    )
    acquire = request(
        templates=(
            replace(
                base,
                events=(
                    replace(
                        drive,
                        start_us=np.zeros(2),
                        duration_us=np.full(2, 0.128),
                        gain=np.full(2, -0.002),
                        envelopes=(waveform,),
                    ),
                    replace(readout, start_us=np.full(2, 0.140)),
                ),
            ),
        )
    )
    try:
        draw(source, acquire)
        numeric = source.model_facts(snapshot()).source
        first_usage = numeric.gate_usage
        assert first_usage
        first_paths = source.evaluations[0].paths
        draw(source, replace(acquire, sequence=8))
        assert source.evaluations[0].paths == first_paths
        assert numeric.gate_usage == first_usage
        assert cache.hits == 1
        assert cache.puts == 1, (
            "unchanged results must retain their population preparation"
        )
    finally:
        source.close()


@pytest.mark.parametrize("clear", [False, True])
def test_repeated_content_does_not_inflate_source_amortization(
    clear, delayed_readout_template
):
    import numpy as np
    from zcu_tools.core.simulation import PulseEnvelope

    base = delayed_readout_template
    drive, readout = base.events
    times = np.linspace(0.0, 0.128, 129)
    waveform = PulseEnvelope(
        tuple(times),
        tuple(np.exp(-0.5 * ((times - 0.064) / 0.032) ** 2).astype(complex)),
    )
    acquire = request(
        templates=(
            replace(
                base,
                events=(
                    replace(
                        drive,
                        start_us=np.zeros(2),
                        duration_us=np.full(2, 0.128),
                        gain=np.full(2, -0.002),
                        envelopes=(waveform,),
                    ),
                    replace(readout, start_us=np.full(2, 0.140)),
                ),
            ),
        )
    )
    source = FTFSource(settings(), DeviceManager())
    try:
        draw(source, acquire)
        numeric = source.model_facts(snapshot()).source
        first = numeric.gate_usage
        first_paths = source.evaluations[0].paths
        assert first and first_paths.segments_by_kernel.get("shaped_sambe") == 2
        for sequence in range(8, 20):
            if clear:
                source.preparation_cache.clear()
            draw(source, replace(acquire, sequence=sequence))
            assert numeric.gate_usage == first, (
                "one content contributes its rows only once"
            )
            assert source.evaluations[0].paths == first_paths
    finally:
        source.close()


def test_equal_content_acquires_skip_adc_preparation_and_keep_streams_exact():
    source = FTFSource(settings(), DeviceManager())
    reference = FTFSource(settings(), DeviceManager())
    try:
        first = expect_statistics(draw(source))
        later = replace(request(), sequence=8)
        second_delivery = draw(source, later)
        assert isinstance(second_delivery, IntegratedRound)
        second = expect_statistics(second_delivery)
        assert source.readout_cache.hits == 0, "equal content must skip ADC preparation"
        assert source.preparation_cache.builds == 1
        assert source.preparation_cache.hits == 1
        expected_delivery = draw(reference, later)
        assert isinstance(expected_delivery, IntegratedRound)
        expected = expect_statistics(expected_delivery)
        assert first.channels[0].sums.tobytes() != second.channels[0].sums.tobytes()
        assert second.channels[0].sums.tobytes() == expected.channels[0].sums.tobytes()
        assert second.state_counts.tobytes() == expected.state_counts.tobytes()
        assert source.evaluations[0].acquire_sequence == 8
        assert source.evaluations[0].round_index == 0
        assert (
            second_delivery.raw.materialize(later.reps, later.avg_level)[0].tobytes()
            == expected_delivery.raw.materialize(later.reps, later.avg_level)[
                0
            ].tobytes()
        )
    finally:
        source.close()
        reference.close()


@pytest.mark.parametrize(
    "change",
    [
        "contents",
        "envelope",
        "snapshot",
        "receiver",
        "tolerance",
        "revision",
        "engine",
        "mode",
        "layout",
        "reps",
    ],
)
def test_changed_preparation_inputs_miss(change, monkeypatch):
    source = FTFSource(settings(), DeviceManager())
    cleanup = ExitStack()
    try:
        draw(source)
        assert hasattr(source, "preparation_cache"), (
            "source must own bounded preparations"
        )
        acquire, snap = request(), snapshot()
        if change in ("contents", "envelope"):
            t = acquire.templates[0]
            event = t.events[0]
            if change == "contents":
                # Borrowed C1 arrays can change in place, even with the same record identity.
                event.phase_deg[0] += 3
            else:
                env = event.envelopes[0]
                assert env is not None
                event = replace(event, envelopes=(replace(env, values=(1j, 1j)),))
            acquire = replace(acquire, templates=(replace(t, events=(event,)),))
        elif change == "snapshot":
            snap = replace(snap, model_identity="changed")
        elif change == "receiver":
            source.readout = replace(
                source.readout,
                receivers=tuple(
                    replace(r, gain=2 * r.gain) for r in source.readout.receivers
                ),
            )
        elif change == "tolerance":
            n = source.settings.numerics
            source.settings = source.settings.model_copy(
                update={
                    "numerics": n.model_copy(
                        update={
                            "solver": n.solver.model_copy(
                                update={"rtol": n.solver.rtol * 2}
                            )
                        }
                    )
                }
            )
        elif change == "revision":
            monkeypatch.setattr(replay, "BACKEND_VERSION", "changed-backend")
        elif change == "engine":
            from zcu_tools.simulate.evolution import Engine

            cleanup.callback(source.engine.close)
            source.engine = Engine(1, 1024**2)
        elif change == "mode":
            acquire = request("decimated")
        elif change == "layout":
            acquire = replace(
                acquire,
                point_dims=(1, 2),
                point_names=("outer", "inner"),
                loop_dims=(3, 1, 2),
            )
        else:
            acquire = replace(acquire, reps=4, loop_dims=(4, 2))
        draw(source, acquire, snap)
        assert source.preparation_cache.builds == 2
        assert source.preparation_cache.hits == 0
    finally:
        source.close()
        cleanup.close()


def test_lru_budget_close_and_failed_preparation_release_residents():
    source = FTFSource(settings(), DeviceManager())
    try:
        draw(source)
        assert hasattr(source, "preparation_cache"), "preparations need a byte budget"
        cache = source.preparation_cache
        size = cache.resident_bytes
        assert size > 0
        cache.max_bytes = size * 2 + 1024
        other = replace(request(), reps=4, loop_dims=(4, 2))
        draw(source, other)
        draw(source)  # Refresh first key.
        third = replace(request(), reps=5, loop_dims=(5, 2))
        draw(source, third)
        assert cache.resident_bytes <= cache.max_bytes
        builds = cache.builds
        draw(source, other)
        assert cache.builds == builds + 1, "least recently used key must be evicted"
        invalid = replace(
            request(), templates=(replace(request().templates[0], events=()),)
        )
        with pytest.raises(ValueError, match="readout"):
            draw(source, invalid)
        assert cache.resident_bytes == 0
        assert source.readout_cache.resident_bytes == 0
        draw(source)
        source.close()
        assert cache.resident_bytes == 0
        with pytest.raises(RuntimeError, match="closed"):
            source.open_acquire(request())
    finally:
        source.close()


def test_preparation_hits_reemit_accuracy_warning_and_keep_numeric_identity():
    from zcu_tools.simulate.evolution import Engine
    from zcu_tools.simulate.evolution.kernels import REGISTRY
    from zcu_tools.simulate.evolution.kernels.t4 import T4

    class ExceedingIdle(T4):
        def apply(self, prepared, key_index, table, rows, states, context):
            result = super().apply(prepared, key_index, table, rows, states, context)
            result[:] = True
            return result

    kernels = tuple(ExceedingIdle() if k.name == "t4" else k for k in REGISTRY)
    source = FTFSource(
        settings(), DeviceManager(), engine=Engine(1, 1024**2, kernels=kernels)
    )
    try:
        with pytest.warns(RuntimeWarning, match="segment accuracy check exceeded"):
            draw(source)
        paths = source.evaluations[0].paths
        with pytest.warns(RuntimeWarning, match="segment accuracy check exceeded"):
            draw(source, replace(request(), sequence=8))
        assert source.evaluations[0].paths == paths
        assert source.evaluations[0].acquire_sequence == 8
    finally:
        source.close()


def test_public_acquire_stop_reentry_and_delayed_raw_survive_preparation_hits():
    from zcu_tools.experiment.ftf import ftf
    from zcu_tools.program.v2 import ModularProgramV2, ProgramV2Cfg
    from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
    from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

    env = ftf(settings())
    try:
        prog = ModularProgramV2(
            env.soccfg,
            ProgramV2Cfg(reps=3, rounds=3),
            modules=[
                PulseReadoutCfg(
                    gen_ch=0,
                    ro_ch=0,
                    nqz=1,
                    freq=6000,
                    gain=0.1,
                    waveform=ConstWaveformCfg(length=0.02),
                    ro_length=0.02,
                ).build("readout")
            ],
        )
        observed = []

        def hook(count, data, stop):
            observed.append(count)
            with pytest.raises(RuntimeError, match="active|reentr"):
                prog.acquire(env.soc, progress=False)
            stop.set()

        prog.acquire(env.soc, progress=False, round_hook=hook)
        first_request = env.soc.binding.last_request
        assert first_request is not None
        raw = prog.get_raw()
        assert raw is not None
        first_raw = raw[0].copy()
        prog.acquire(env.soc, progress=False, round_hook=hook)
        assert observed == [1, 1]
        second_request = env.soc.binding.last_request
        assert second_request is not None
        assert second_request.sequence == first_request.sequence + 1
        assert raw[0].tobytes() == first_raw.tobytes()
        again = prog.get_raw()
        assert again is not None
        assert again[0].shape == first_raw.shape
    finally:
        env.devices.close_all_devices()
