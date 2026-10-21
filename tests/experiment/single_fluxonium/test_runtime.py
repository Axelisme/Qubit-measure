"""Public source lifecycle, live snapshots and content-addressed plan reuse."""

import importlib
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.core.simulation import AcquireRequest, PulseEnvelope, RoundRequest
from zcu_tools.device import DeviceManager, FakeDevice

from ._inputs import settings, template


def test_source_refreshes_model_and_plan_for_live_dc_and_waveform_content() -> None:
    runtime = importlib.import_module("zcu_tools.experiment.single_fluxonium.runtime")
    config = settings()
    devices = DeviceManager()
    bias = FakeDevice(fast_mode=True, value_unit="A")
    devices.register_device(config.dc_bias.source.name, bias)
    source = runtime.SingleFluxoniumSource(config, devices)
    try:
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
    finally:
        source.close()
        devices.close_all_devices()
    with pytest.raises(RuntimeError, match="closed"):
        source.model_facts(first)


def test_unconnected_acquisition_fails_without_claiming_a_round_result() -> None:
    runtime = importlib.import_module("zcu_tools.experiment.single_fluxonium.runtime")
    config = settings()
    devices = DeviceManager()
    bias = FakeDevice(fast_mode=True, value_unit="A")
    devices.register_device(config.dc_bias.source.name, bias)
    source = runtime.SingleFluxoniumSource(config, devices)
    request = AcquireRequest("integrated", (1,), 0, (), 123, 0, "test")
    try:
        with pytest.raises(NotImplementedError, match="single-cavity readout"):
            source.open_acquire(request)
        with pytest.raises(NotImplementedError, match="single-cavity readout"):
            source.execute(RoundRequest(request, source.snapshot(), 0))
    finally:
        source.close()
        devices.close_all_devices()
