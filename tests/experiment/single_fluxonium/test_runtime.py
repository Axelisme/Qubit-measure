"""Public source lifecycle, live snapshots and content-addressed plan reuse."""

from __future__ import annotations

import importlib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from zcu_tools.core.simulation import (
    AcquireRequest,
    PulseEnvelope,
    ReadoutWindow,
    RoundRequest,
)
from zcu_tools.device import DeviceManager, FakeDevice

from ._inputs import settings, template

if TYPE_CHECKING:
    from zcu_tools.experiment.single_fluxonium.runtime import SingleFluxoniumSource
    from zcu_tools.resources.single_fluxonium.schema import SingleFluxoniumSettings


@contextmanager
def source_environment() -> Generator[
    tuple[SingleFluxoniumSource, FakeDevice, SingleFluxoniumSettings]
]:
    runtime = importlib.import_module("zcu_tools.experiment.single_fluxonium.runtime")
    config = settings()
    devices = DeviceManager()
    try:
        bias = FakeDevice(fast_mode=True, value_unit="A")
        devices.register_device(config.dc_bias.source.name, bias)
        source = runtime.SingleFluxoniumSource(config, devices)
        try:
            yield source, bias, config
        finally:
            source.close()
    finally:
        devices.close_all_devices()


def test_source_refreshes_model_and_plan_for_live_dc_and_waveform_content() -> None:
    with source_environment() as (source, bias, config):
        first = source.snapshot()
        facts = source.model_facts(first)
        assert facts.model.dc_flux == 0.5
        assert facts.source.initial_density.shape == (4, 4)
        model = facts.source.model((0,))
        _, vectors = facts.model.eigensystem(0.5)
        charge = vectors.conj().T @ facts.model.charge_operator() @ vectors
        np.testing.assert_allclose(
            model.drives[0].in_phase_ghz, 0.1 * charge, atol=1e-12
        )
        np.testing.assert_allclose(
            model.drives[0].quadrature_ghz, 0.02 * charge, atol=1e-12
        )
        assert source.model_facts(first) is facts
        original = template()
        planned = source.plan_template(first, original)
        assert source.plan_template(first, original) is planned
        # Same metadata and support, different samples must not reuse the plan.
        changed_event = replace(
            original.events[1], envelopes=(PulseEnvelope((0, 0.008), (0j, 1j)),)
        )
        changed = replace(
            original, events=(original.events[0], changed_event, original.events[2])
        )
        new_plan = source.plan_template(first, changed)
        assert new_plan is not planned
        assert new_plan.batch.flux_waveform is not None
        np.testing.assert_allclose(new_plan.batch.flux_waveform.flux, 0.5)
        assert planned.batch.flux_waveform is not None
        assert np.max(planned.batch.flux_waveform.flux) > 0.59
        # Borrowed arrays are reread, not keyed only by template identity.
        original.events[1].gain[:] = 0
        zero = source.plan_template(first, original)
        assert zero.batch.flux_waveform is not None
        np.testing.assert_allclose(zero.batch.flux_waveform.flux, 0.5)
        bias.set_value(0.2)
        second = source.snapshot()
        assert second.values == ((config.dc_bias.source.name, 0.2),)
        moved = source.model_facts(second)
        assert moved.model.dc_flux == pytest.approx(0.52)
        assert moved is not facts
        assert not np.allclose(moved.model.energies_ghz, facts.model.energies_ghz)
        shifted = source.plan_template(second, original)
        assert shifted.batch.flux_waveform is not None
        np.testing.assert_allclose(shifted.batch.flux_waveform.flux, 0.52)
        bias.set_value(2)
        with pytest.raises(ValueError, match="limits"):
            source.snapshot()
    with pytest.raises(RuntimeError, match="closed"):
        source.model_facts(first)


def preflight_request(compiled=None):
    return AcquireRequest(
        "integrated",
        (1, 4),
        0,
        (ReadoutWindow(0, 0, 4, 1000, (0.018,)),),
        123,
        0,
        f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
        point_dims=(4,),
        point_names=("sweep",),
        templates=(template() if compiled is None else compiled,),
    )


@pytest.mark.parametrize(
    ("part", "field", "value"),
    [
        ("template", "points", [1, 2]),
        ("drive", "start_us", 0.003),
        ("drive", "duration_us", 0.004),
        ("drive", "frequency_mhz", 300),
        ("drive", "phase_deg", 45),
        ("drive", "gain", 0.1),
        ("drive", "active", False),
        ("drive", "envelope_index", 1),
        ("flux", "gain", 0),
        ("readout", "trigger_us", 0.019),
        ("readout", "frequency_mhz", 6999),
        ("readout", "phase_deg", 70),
    ],
)
def test_preflight_rereads_borrowed_arrays_like_a_fresh_source(part, field, value):
    compiled = template()
    drive = replace(
        compiled.events[0], envelopes=(None, PulseEnvelope((0, 0.006), (0j, 1j)))
    )
    compiled = replace(compiled, events=(drive, *compiled.events[1:]))
    request = preflight_request(compiled)
    with source_environment() as (source, _, _):
        snapshot = source.snapshot()
        first = source.preflight(request, snapshot)
        detached = asdict(first.plans[0][0])
        repeated = source.preflight(replace(request, sequence=1), snapshot)
        np.testing.assert_equal(asdict(repeated.plans[0][0]), asdict(first.plans[0][0]))
        target = {
            "template": compiled,
            "drive": compiled.events[0],
            "flux": compiled.events[1],
            "readout": compiled.readouts[0],
        }[part]
        getattr(target, field)[:] = value
        changed = source.preflight(request, snapshot)
        with source_environment() as (fresh, _, _):
            expected = fresh.preflight(request, snapshot)
        np.testing.assert_equal(
            asdict(changed.plans[0][0]), asdict(expected.plans[0][0])
        )
        # The old detached plan remains safe for callers retaining it.
        np.testing.assert_equal(asdict(first.plans[0][0]), detached)


def test_preflight_refreshes_snapshot_settings_and_immutable_envelopes():
    request = preflight_request()
    with source_environment() as (source, bias, config):
        first = source.preflight(request, source.snapshot())
        bias.set_value(0.2)
        moved = source.preflight(request, source.snapshot())
        moved_waveform = moved.plans[0][0].batch.flux_waveform
        assert moved_waveform is not None
        assert moved_waveform.flux[0, 0] == pytest.approx(0.52)
        line = config.flux_line
        assert line is not None
        source.settings = config.model_copy(
            update={"flux_line": line.model_copy(update={"flux_phi0_per_gain": 0.4})}
        )
        calibrated = source.preflight(request, source.snapshot())
        assert calibrated.plans[0][0].batch.flux_waveform is not None
        assert moved.plans[0][0].batch.flux_waveform is not None
        assert np.max(calibrated.plans[0][0].batch.flux_waveform.flux) > np.max(
            moved.plans[0][0].batch.flux_waveform.flux
        )
        with pytest.raises(ValueError, match="settings changed"):
            source.prepare_readouts(first)
        original = request.templates[0]
        changed = replace(
            original.events[0], envelopes=(PulseEnvelope((0, 0.006), (0j, 1j)),)
        )
        request = replace(
            request,
            templates=(replace(original, events=(changed, *original.events[1:])),),
        )
        planned = source.preflight(request, source.snapshot())
        np.testing.assert_array_equal(
            planned.plans[0][0].batch.tracks[0].envelopes[0].samples, [0j, 1j]
        )


def test_preflight_keeps_receiver_order_calibration_and_layout_validation():
    with source_environment() as (source, _, config):
        receiver = config.readout.adc
        receivers = (receiver.model_copy(update={"ro_ch": 3}), receiver)
        original = template()
        windows = (replace(original.readouts[0], channel=3), original.readouts[0])
        request = replace(
            preflight_request(replace(original, readouts=windows)),
            readouts=(
                ReadoutWindow(3, 0, 4, 1000, (0.018,)),
                ReadoutWindow(0, 0, 4, 1000, (0.018,)),
            ),
        )
        snapshot = source.snapshot()
        first = source.preflight(request, snapshot, receivers=receivers)
        assert tuple(receiver.ro_ch for receiver in first.receivers) == (3, 0)
        changed = (receivers[0].model_copy(update={"gain": 2.0}), receiver)
        calibrated = source.preflight(request, snapshot, receivers=changed)
        assert calibrated.receivers[0].gain == 2.0
        # A hit must not return the preceding request or sampling identity.
        next_request = replace(request, sequence=7, root_seed=456)
        assert (
            source.preflight(next_request, snapshot, receivers=changed).request
            is next_request
        )
        with pytest.raises(ValueError, match="complete calibrated receiver set"):
            source.preflight(request, snapshot, receivers=tuple(reversed(receivers)))
        conflict = replace(
            windows[1],
            trigger_us=np.repeat(windows[1].trigger_us, 2, axis=1),
            frequency_mhz=np.repeat(windows[1].frequency_mhz, 2, axis=1),
            phase_deg=np.repeat(windows[1].phase_deg, 2, axis=1),
            generator_channels=np.repeat(windows[1].generator_channels, 2, axis=1),
        )
        other = replace(
            original,
            points=np.array([1, 2], dtype=np.int64),
            readouts=(windows[0], conflict),
        )
        with pytest.raises(ValueError, match="ADC 0: template 1 readout layout"):
            source.preflight(
                replace(request, templates=(*request.templates, other)),
                snapshot,
                receivers=receivers,
            )
        windows[0].generator_channels[:] = 2
        with pytest.raises(ValueError, match="bound readout generator"):
            source.preflight(request, snapshot, receivers=receivers)


@pytest.mark.sim
def test_preflight_preparation_applies_each_receivers_current_calibration():
    with source_environment() as (source, _, config):
        original = template()
        compiled = replace(
            original,
            points=np.array([0, 1], dtype=np.int64),
            events=(original.events[-1],),
            readouts=(replace(original.readouts[0], channel=3), original.readouts[0]),
        )
        request = replace(
            preflight_request(compiled),
            loop_dims=(1, 2),
            point_dims=(2,),
            readouts=(
                ReadoutWindow(3, 0, 4, 1000, (0.018,)),
                ReadoutWindow(0, 0, 4, 1000, (0.018,)),
            ),
        )
        receiver = config.readout.adc
        receivers = (receiver.model_copy(update={"ro_ch": 3}), receiver)
        snapshot = source.snapshot()
        first = source.prepare_readouts(
            source.preflight(request, snapshot, receivers=receivers)
        )
        source.preflight(request, snapshot, receivers=receivers)
        changed = (receivers[0].model_copy(update={"gain": 2.0}), receiver)
        second = source.prepare_readouts(
            source.preflight(request, snapshot, receivers=changed)
        )
        assert tuple(channel.channel for channel in second.channels) == (3, 0)
        assert np.max(np.abs(first.channels[0].conditional.sums)) > 0
        np.testing.assert_array_equal(
            second.channels[0].conditional.sums, 2 * first.channels[0].conditional.sums
        )
        np.testing.assert_array_equal(
            second.channels[1].conditional.sums, first.channels[1].conditional.sums
        )
        reordered = replace(
            request,
            readouts=tuple(reversed(request.readouts)),
            templates=(replace(compiled, readouts=tuple(reversed(compiled.readouts))),),
        )
        reversed_prepared = source.prepare_readouts(
            source.preflight(reordered, snapshot, receivers=tuple(reversed(changed)))
        )
        assert tuple(channel.channel for channel in reversed_prepared.channels) == (
            0,
            3,
        )
        np.testing.assert_array_equal(
            reversed_prepared.channels[1].conditional.sums,
            second.channels[0].conditional.sums,
        )


def test_builder_binds_control_generators_and_live_dc_with_single_cavity_readout() -> (
    None
):
    runtime = importlib.import_module("zcu_tools.experiment.single_fluxonium.runtime")
    config = settings()
    environment = runtime.single_fluxonium(config)
    compiled = replace(template(), points=np.array([0, 1], dtype=np.int64))
    request = AcquireRequest(
        "integrated",
        (1, 2),
        0,
        (ReadoutWindow(0, 0, 4, 1000, (0.018,)),),
        123,
        0,
        f"PCG64/SeedSequence-v2/numpy-{np.__version__}",
        point_dims=(2,),
        point_names=("sweep",),
        templates=(compiled,),
    )
    try:
        device = environment.create_device("FakeDevice", config.dc_bias.source.name)
        assert device.get_info().value_unit == "A"
        assert device.get_info().value == config.dc_bias.source.initial
        device.set_value(0.2)
        snapshot = environment.soc.binding.capture_snapshot()
        assert snapshot.values == ((config.dc_bias.source.name, 0.2),)
        # Admission checks source wiring only; it does not execute an acquire.
        assert {gen for gen, _ in environment.soc.routes} == {0, 1, 2}
        environment.soc.install_request(request)
        from zcu_tools.program.v2.simulation import IntegratedRound

        result = environment.soc.binding.execute(RoundRequest(request, snapshot, 0))
        if not isinstance(result, IntegratedRound):
            raise AssertionError("builder must deliver a physical integrated round")
        np.testing.assert_array_equal(
            result.statistics.state_counts.sum(axis=-1), [1, 1]
        )
    finally:
        environment.soc.release_request()
        environment.devices.close_all_devices()
