"""A compiled readout lookup has the same physical IQ as matching linear points."""

from math import prod

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.readout_plan import (
    ReadoutDrive,
    map_readout_events,
    readout_contents,
)
from zcu_tools.program.v2 import (
    PulseReadout,
    PulseReadoutCfg,
    SweepCfg,
    TimelineBuilder,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.simulation import SimulationBinding, SimulationSoc
from zcu_tools.simulate.adc import (
    ProjectionReadout,
    ReadoutBatch,
    ReadoutModel,
    Receiver,
    sample_traces,
)
from zcu_tools.simulate.boundary import (
    EnvironmentSnapshot,
    RoundRequest,
    RoundSignals,
    SamplingIdentity,
)

from tests.program._profiles import load_soccfg


def test_table_readout_iq_matches_linear_at_each_compiled_frequency():
    soccfg = load_soccfg(n_gens=1, n_readouts=1)
    model = ReadoutModel(
        identity="one-cavity-for-frequency-sweep",
        state_labels=("ground",),
        frequencies_ghz=np.array([[6.9]]),
        reference_frequency_ghz=6.9,
        kappa_internal_per_ns=np.array([0.001]),
        kappa_external_per_ns=np.array([0.01]),
        rates_per_ns=np.zeros((1, 1)),
        mismatch_phase_rad=np.zeros(1),
        receiver=Receiver(gain=20.0, phase_rad=0.2, time_of_flight_ns=11.0),
    )

    def execute(request: RoundRequest) -> RoundSignals:
        acquire = request.acquire
        (template,) = acquire.templates
        (window,) = template.readouts
        contents = readout_contents(
            template,
            ReadoutDrive(gen_ch=0, input_per_gain=3.0),
            window,
            {},
            time_base=acquire.time_base,
        )
        mapped = map_readout_events((contents,), model.reference_frequency_ghz)
        batch = ReadoutBatch(
            model, mapped.events, mapped.inverse, output_order=mapped.output_order
        )
        return sample_traces(
            ProjectionReadout.unprojected(np.ones((*acquire.point_dims, 1))),
            {window.channel: batch},
            reps=acquire.reps,
            identity=SamplingIdentity(
                acquire.root_seed, (acquire.sequence, request.index, 0)
            ),
        )

    binding = SimulationBinding(
        lambda: EnvironmentSnapshot("one-cavity-v1", model.identity),
        execute,
        backend_version="one-cavity-v1",
        seed=17,
    )
    soc = SimulationSoc(soccfg, binding, routes=((0, 0),))
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=2,
        freq=6900.0,
        gain=0.25,
        phase=31.0,
        waveform=ConstWaveformCfg(length=0.22),
        ro_length=0.09,
        trig_offset=0.056,
    )

    def acquire_at(
        sweep: list[float] | SweepCfg,
    ) -> tuple[list[float], NDArray[np.float64]]:
        builder = TimelineBuilder(soccfg, reps=1)
        freq = (
            builder.add_sweep("freq", sweep)
            if isinstance(sweep, SweepCfg)
            else builder.add_table_sweep("freq", sweep)
        )
        builder.emit(PulseReadout("readout", readout, freq=freq))
        (iq,) = builder.build().acquire_decimated(soc, progress=False)
        request = binding.last_request
        assert request is not None
        actual = [
            request.materialize_point(index).events[0].frequency_mhz
            for index in range(prod(request.point_dims))
        ]
        return actual, iq

    # Deliberately non-uniform and unsorted so swapped table rows do not pass.
    requested = [6896.0, 6901.25, 6899.0, 6904.5]
    frequencies, table = acquire_at(requested)
    assert len(set(frequencies)) == len(requested)
    assert table.shape[0] == len(requested)
    for index, frequency in enumerate(requested):
        (actual,), linear = acquire_at(
            SweepCfg(start=frequency, stop=frequency, step=0.0, expts=1)
        )
        assert frequencies[index] == actual
        actual_iq = table[index].reshape(-1, 2)
        assert np.any(actual_iq != 0.0)
        np.testing.assert_allclose(actual_iq, linear.reshape(-1, 2), atol=1e-9, rtol=0)
