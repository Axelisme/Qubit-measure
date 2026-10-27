"""Single-cavity physical model and ADC preparation; sampling stays with ADC."""

from dataclasses import replace
from math import prod

import numpy as np

from zcu_tools.core.simulation import AcquireRequest
from zcu_tools.experiment.readout_plan import local_readout_pulses
from zcu_tools.resources.single_fluxonium import SingleFluxoniumSettings
from zcu_tools.simulate.adc import (
    BackgroundBandEdge,
    BackgroundBandEdges,
    BackgroundRipple,
    BackgroundRoughness,
    ChannelReadout,
    ConditionalReadout,
    ConditionalSums,
    ReadoutBatch,
    ReadoutModel,
    Receiver,
)
from zcu_tools.simulate.evolution import ModelSource
from zcu_tools.simulate.fluxonium import DispersiveResonator, SingleFluxoniumModel

from .template import SingleFluxoniumPlan


def dispersive_resonator(settings: SingleFluxoniumSettings) -> DispersiveResonator:
    """Convert validated cavity settings for Purcell loss and dressed readout.

    Frequencies and charge coupling remain in GHz. The returned total linewidth
    is angular inverse ns, converted from internal plus external MHz linewidths.
    Thermal occupation is passed through, including None for equilibrium.
    No model construction, I/O or mutation occurs.
    """
    resonator = settings.readout.resonator
    return DispersiveResonator(
        resonator.frequency_ghz,
        resonator.charge_coefficient_ghz,
        2
        * np.pi
        * (resonator.kappa_external_mhz + resonator.kappa_internal_mhz)
        / 1000,
        resonator.thermal_occupation,
    )


def readout_model(
    settings: SingleFluxoniumSettings,
    model: SingleFluxoniumModel,
    source: ModelSource,
) -> ReadoutModel:
    """Build one neutral cavity in the model's retained DC energy-state order.

    settings/model/source must describe the same fixed DC point. Frequencies use
    the model owner's dressed Phi0 shifts, and directional rates use its DC bath.
    No receiver calibration, pulse, sampling or evolution is performed. Numeric
    model/dressed-labeling failures propagate. The returned model owns its arrays.
    """
    resonator = settings.readout.resonator
    secular = source.model((0,))
    frequencies = model.readout_frequencies_ghz(dispersive_resonator(settings))
    return ReadoutModel(
        secular.identity,
        tuple(str(index) for index in range(model.retained_levels)),
        frequencies[:, None],
        resonator.frequency_ghz,
        np.array([2 * np.pi * resonator.kappa_internal_mhz / 1000]),
        np.array([2 * np.pi * resonator.kappa_external_mhz / 1000]),
        secular.rates_per_ns,
        mismatch_phase_rad=np.array([resonator.mismatch_phase_rad]),
    )


def prepare_readout(
    settings: SingleFluxoniumSettings,
    physical: ReadoutModel,
    plans: tuple[SingleFluxoniumPlan, ...],
    request: AcquireRequest,
) -> tuple[ChannelReadout, ReadoutBatch]:
    """Assemble one ADC's conditional sums and raw inputs for the whole request.

    plans align with request.templates and share physical's DC state order.
    settings supplies the one receiver; PSD converts to per-sample covariance
    using the declared sample rate. Absolute grid timing and pulse-local phase
    conversion remain owned by readout_plan. ADC computes all conditional sums.
    Returned content indices use request point-major order, not template order.
    Unequal sample rates across templates raise ValueError. ADC shape/solver
    errors propagate; no sampling, normalization or QICK averaging occurs here.
    """
    adc = settings.readout.adc
    rate = request.templates[0].readouts[0].sample_rate_mhz
    if any(
        template.readouts[0].sample_rate_mhz != rate for template in request.templates
    ):
        raise ValueError("single-cavity templates require one ADC sample rate")
    receiver = Receiver(
        adc.gain,
        float(np.deg2rad(adc.phase_deg)),
        np.array(adc.offset_adc),
        np.array(adc.noise_psd_adc2_ns) / (1000 / rate),
        time_of_flight_ns=adc.time_of_flight_ns,
        ripple_depth=adc.background.ripple_depth,
        ripple_period_ghz=adc.background.ripple_period_mhz / 1000,
        ripple_phase_rad=float(np.deg2rad(adc.background.ripple_phase_deg)),
        slope_db_per_ghz=adc.background.slope_db_per_ghz,
        pivot_ghz=adc.background.pivot_ghz,
        extra_ripples=tuple(
            BackgroundRipple(w.depth, w.period_mhz, w.phase_deg)
            for w in adc.background.extra_ripples
        ),
        roughness=None
        if adc.background.roughness is None
        else BackgroundRoughness(
            adc.background.roughness.amplitude,
            adc.background.roughness.correlation_mhz,
            adc.background.roughness.seed,
        ),
        band_edges=None
        if adc.background.band_edges is None
        else BackgroundBandEdges(
            low=None
            if adc.background.band_edges.low is None
            else BackgroundBandEdge(
                adc.background.band_edges.low.corner_ghz,
                adc.background.band_edges.low.rolloff_db_per_ghz,
            ),
            high=None
            if adc.background.band_edges.high is None
            else BackgroundBandEdge(
                adc.background.band_edges.high.corner_ghz,
                adc.background.band_edges.high.rolloff_db_per_ghz,
            ),
        ),
    )
    reference = plans[0].readout.pulses[0][0].frequency_ghz
    model = replace(physical, receiver=receiver, reference_frequency_ghz=reference)
    solver = ConditionalReadout(
        model,
        rtol=settings.numerics.solver.rtol,
        atol=settings.numerics.solver.atol,
    )
    pulses, grids, conditionals = [], [], []
    content_index = np.empty(prod(request.point_dims), dtype=np.int64)
    for plan in plans:
        content = plan.readout
        local = local_readout_pulses(content.pulses, reference)
        batch = ReadoutBatch(model, local, content.grids, content.inverse)
        conditional = solver.responses(batch)
        content_index[plan.batch.points] = len(pulses) + content.inverse
        pulses.extend(local)
        grids.extend(content.grids)
        conditionals.append(conditional)
    conditional = ConditionalSums(
        np.concatenate([value.sums for value in conditionals]),
        conditionals[0].sample_counts,
    )
    return (
        ChannelReadout(
            adc.ro_ch, conditional, content_index.reshape(request.point_dims), receiver
        ),
        ReadoutBatch(model, tuple(pulses), tuple(grids), content_index),
    )
