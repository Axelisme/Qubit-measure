# Numeric simulation tools

**Last updated:** 2026-10-16，FTF joint circuits and independent numeric tools

`evolution` and `adc` are independent numeric tools. Neither imports QICK,
DeviceManager, resources, GUI, YAML or an environment. The existing fluxonium
prediction tools remain separate. ADR-0072 defines the package dependency rules.

## Evolution and explicit populations

Import records and functions from `zcu_tools.simulate.evolution`.
`NumericModel` owns a fixed `Basis`, static Hamiltonian and source quadratures.
Optional numeric callbacks provide additive time-dependent Hamiltonians and baths
in that same basis. Hamiltonian coefficients are H/h in GHz, time is ns, phase
is radians, and Lindblad operators include their rates in ns**(-1/2).
The solver applies the Hamiltonian factor `-2*pi*1j`; source amplitudes have no
hidden factor of one half or per-waveform normalization.

`Evolution.evolve` requires an explicit `DensityState` on every call. It returns
only the final density, not an internal-step trajectory. Pass the previous result
and the same full absolute `ControlSequence` when continuing a shot. A new shot
must receive its own initial state. Instances do not remember quantum states.

Model event times partition bath discontinuities alongside source events.
The solver rounds all event coordinates to the nearest 2 ns from origin zero,
with ties to even. It unions the coordinates before forming intervals. It never
rounds interval lengths independently. Envelopes stay continuous in pulse-local
coordinates across other sources' events. `Envelope.sampled` preserves the
original arbitrary-waveform axis and signed complex values.

Within each joint interval the solver uses a static matrix exponential, adaptive
DOP853, or an exact-frequency period plus tail. Integer propagator powers use
binary multiplication. Only flat, exactly commensurate closed controls qualify
for period reuse. Dissipative evolution solves the full Lindblad equation.
Solver tolerances control numerical integration, not basis convergence or the
physical validity of a model. Solver output is never repaired by clipping
negative density eigenvalues or renormalizing the trace.

Call `populations(state, target_basis)` explicitly before readout. Basis vectors
are ordered columns in common ambient coordinates. The result preserves target
state labels and can be recorded independently of ADC generation.

## ADC readout

Import from `zcu_tools.simulate.adc`. Construct `ReadoutModel` with joint-state
labels, absolute resonator frequencies, two sets of resonator decay rates and a
directional transition matrix indexed `[target, source]`. Supply `PopulationInput`
with exactly the same state order. The ADC does not convert density matrices.

`ADC.response` solves populations and state-conditioned first field moments for
all initial joint states together. Two resonators share reciprocal co-located
two-port radiative damping through `v*v.T`, where `v=sqrt(kappa_external/2)`.
The outgoing forward field is `b_in-v.T*alpha`. This retains correlations between
state and field. Substituting a population-averaged frequency does not implement
this model. Each pulse has finite support, so the same equations cover ringup
and ringdown. `SampleGrid` supplies physical lookback times and demodulation.
Populations refer to the earliest readout pulse start, or zero with no pulses.
Earlier samples have zero physical field and do not advance the initial mixture.
Receiver offset and noise still apply to those samples.

`ADC.sample_windows` shares one initial-state draw across several per-execution
windows, while receiver noise remains independent across windows.
`ADC.sample` computes these conditional means once and draws one initial joint
state per shot, then adds independent per-sample receiver noise. It does not draw
jump times or solve an ODE per shot. The result includes float64 traces and their
exact discrete sums from one realization. Offset is added once per sample.
A zero or singular positive-semidefinite 2x2 I/Q noise covariance is valid.
`SamplingIdentity` selects a local PCG64 stream, without changing global RNG state.

Environment code owns unit conversion from QICK us/MHz/degrees to ns/GHz/radians
and composes evolution, explicit populations and ADC. Program adapters still own
channel order, QICK buffer shapes, nearest-even int64 conversion and acquisition
normalization. [`evolution`](evolution/README.md) also owns optional offline
Intel OpenCL U tables, checked numeric files and persistent exact density
records. GPU construction is offline only. Runtime fixes published versions at
the environment-selected boundary and directly solves uncovered legal requests.
There is no backend registry or automatic runtime table construction.

## FTF circuits and noise

[`ftf`](ftf/README.md) builds the fixed-coordinate F1-SQUID-F2 joint model with
full charge couplings, native-before-projection n², directional multilevel baths
and parent virtual readout transitions. `spectra` owns shared scalar/array loss spectra in angular GHz/ns/K;
`baths` uses them for joint directional rates, while fluxonium coherence combines
both directions for equilibrium T1. Default Q models and caller-supplied Q,
impedance and admittance options retain their existing definitions;
`dephasing` owns both the shared Notebook pure-rate formulas and CP multilevel
Markov dephasing. `lines.FilteredFlux` applies causal source delay/lowpass before
loop mixing and retains passive tails within a shot. These tools do not read
YAML, drivers or program Modules. `experiment.ftf` composes them for acquisition.
