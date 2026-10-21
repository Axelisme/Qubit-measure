# Single-fluxonium settings guide

**Readers:** operator, framework-dev. Read the single-fluxonium settings' units and approximation boundaries.

## Files and wiring

`default.yaml` uses `zcu.single-fluxonium` version 1.
`entry-template.yaml` uses `zcu.entry-template` version 1.0.
Q1 uses microwave generator 1 and fast-flux generator 2.
R1 uses readout generator 0 and ADC 0.
`flux_bias` is the only DC source. The startup script seeds aliases for
`qubit`, `resonator` and `flux_dev` when it creates the entry.

## Circuit and basis

EJ, EC and EL are H/h in GHz. The defaults are 4, 1 and 1 GHz.
Reduced flux is Phi/Phi0. DC flux equals `offset_phi0` plus
`flux_per_unit` times the live device value. Temperature is in K.

`native_cutoff` is the harmonic-oscillator basis dimension.
`levels` is the retained lowest-state space selected at round DC.
Fast-flux Hamiltonians stay in that fixed projected space.
The defaults retain 21 states from 40 oscillator states. This is an accuracy
choice, not a number of displayed states. Check convergence again for different
flux ranges, drives or higher-state observables.

## Source lines

DC units stay A or V for the driver's lifetime. Limits and ramp step use that
same unit. Initial is applied at construction, not before every round.

Microwave calibration multiplies full charge n in GHz per normalized DAC gain.
It includes no hidden n01 normalization or rotating-wave factor.
The program supplies frequency, phase and complex envelope.

Fast-flux calibration is signed Phi0 per gain. Delay and all time constants use
ns. The step response is
`1 - (1-sum(A))*exp(-t/tau_ns) - sum(A*exp(-t/tau_k))` after delay.
A zero base tau gives instantaneous feedthrough. Distortion amplitudes can be
negative. Tails continue within each independent shot until readout.
The planner samples flux at 2 ns plus delayed events; interpolation is linear.
Bath segments use the configured `bath_segment_ns` and quantized 1e-3 Phi0 levels.

## Readout and limitations

Bare cavity frequency and charge coupling use GHz.
External and internal linewidths are kappa/(2*pi) in MHz.
Incident readout amplitude is sqrt(photons/ns) per gain.
Receiver gain maps sqrt(photons/ns) to ADC sample units.
Offset is an I/Q pair. Noise PSD is a two-sided 2-by-2 I/Q covariance in
ADC units squared times ns. Time of flight uses ns. Receiver phase uses degrees;
resonator mismatch phase uses radians.
The model has one physical cavity. The ADC owner supports 1 or 2 cavities;
other cavity counts fail at model construction. Single-cavity ADC callers
explicitly supply one scattering phase, including a one-element zero array
for no mismatch. Receiver I/Q remains two-dimensional.

The model owner supplies energy levels, projected operators, secular losses,
equilibrium 0↔1 Purcell rates and residual-photon dephasing.
The approximations assume weak dispersive coupling and Markov baths.
Acquisition uses the ADC owner's one-cavity conditional response, integrated
statistics, physical raw shots and decimated traces. The density at terminal
readout converts to populations in the DC energy basis. Readout uses DC energy
levels and dressed cavity shifts even when a fast-flux tail remains.

Only terminal readout is supported. Any earlier pulse on the readout generator
raises ValueError before evolution. Cavity-drive Stark shifts, cavity memory
before readout and their backaction are not implemented. A sequence requiring
such a pulse must not be interpreted as a supported measurement.

Solver tolerances control numerical integration, not basis convergence.
Seed selects the root RNG identity. Acquire sequence and software round select
independent streams; raw shots use a stream separate from direct means.
`min_round_duration_s` sets the successful-round pacing floor in seconds.
The current assembly starts no background numerical preparation.
