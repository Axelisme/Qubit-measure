# Six independent fluxonium cavities

**Last updated:** 2026-10-23

**Readers:** framework-dev, lab-author. Build and acquire six independent cavities through the normal simulation SoC.

`six_fluxonium(configuration)` accepts `SixFluxoniumSettings` or a YAML path and
returns `SimulationEnvironment`. The builder creates one addressed `flux_bias`
FakeDevice. Six `SingleFluxoniumSource` instances retain that driver. Each circuit
uses its own DC offset, current-to-flux conversion, oscillator basis and solver.
There is no joint Hilbert space or fast-flux input. Construction does not evolve
states, warm numeric caches or open hardware.

The host calls `close_backend()` after acquisitions drain, then closes its devices.
Backend close leaves the DC driver untouched. Delivered raw sources retain their
noiseless ADC inputs after later acquisitions and backend close.

## Routing and acquisition

Mux generator 0 binds ADC0..5. Microwave generators 1..6 control Q1..Q6.
Pulse generator 13 binds ADC6 for single-tone sweeps. Generators 7..12 are unbound.
At every point, each incident tone selects the nearest bare cavity by frequency.
Each ADC demodulation frequency selects the observed cavity. Distances tied within
1e-9 MHz raise `ValueError`; the tolerance covers floating subtraction only.

Routing follows frequency, not ADC index or tone declaration order. A pulse sweep
can cross all six cavity partitions. Multiple coherent tones assigned to one
cavity enter its existing single-cavity ADC model together. ADCs observing the
same cavity at the same point share state populations and jump histories.
Independent cavities draw separate streams. QICK retains ADC declaration order,
normalization, software-round averaging and companion sums from decimated traces.

A request uses either the pulse route or mux routes. All reads of one ADC at one
point must select the same cavity. Earlier cavity pulses, missing incident tones,
unbound controls and ambiguous frequency assignments fail before evolution.
The backend supports terminal readout only, as the single-model source does.
Pulse ADC6 uses Q1's receiver gain, offset, phase, time of flight and noise PSD.
Its physical cavity and incident-field calibration follow the frequency route.
Q1's seed selects the environment root stream. The largest per-qubit pacing floor
sets the minimum successful-round duration.

## Bounded statistics

The backend partitions only routing patterns present in the request. It prepares
independent cavity populations and ADC responses on those point subsets.
Integrated mode preserves every sampled marginal count. Sorted cumulative-count
boundaries encode the common summary in at most `1 + sum(levels - 1)` bins.
For six 21-level circuits this is 121 bins, rather than `21**6` joint states.
This summary represents marginal moments, not a joint quantum state.
Raw and decimated draws use the ADC owner's physical samplers, not that histogram.
