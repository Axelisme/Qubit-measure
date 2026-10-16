# Fixed-coordinate FTF numerics

**Last updated:** 2026-10-17

`JointParameters` supplies F1/C/F2 numeric circuits and signed full-charge
couplings in GHz. `JointModel` selects fixed local levels at DC, forms their
product space with all g1c/g2c/g12 terms, then selects the lowest M joint states.
Dynamic Hamiltonians stay in this fixed basis. Native n² is formed before local
and joint projection; squared truncated n is not a substitute.

`native_embedding()` maps the retained joint coordinates into physical native
F1-oscillator × coupler-charge × F2-oscillator coordinates, including both local
and coupled eigenvector phases. Authors can compare projectors, coherences and
subspace leakage across truncations without equating energy-index labels.
Oscillator cutoffs pad at the high end; the charge grid pads symmetrically.
Different EC/EL oscillator scales require a separate coordinate transformation.
This explicit diagnostic allocates a tensor only when called; it does not change
runtime propagation or certify basis convergence.

Fluxonium uses fixed oscillator coordinates. The SQUID uses fixed theta and
charge n=-ncut..ncut with H/h=4EC(n-ng)²-EJmax*cos(pi*f)*cos(theta)
+d*EJmax*sin(pi*f)*sin(theta). It does not switch to a flux-dependent shifted
coordinate frame during a pulse. Static spectra agree with effective-EJ SQUID
spectra, but that agreement alone is not a nonadiabatic dynamics guarantee.

`initial_state` returns the coupled DC Gibbs density for the common temperature.
At T=0 exactly degenerate ground states get a uniform mixture. `rates` uses joint
gaps and bath matrix elements for every retained directional transition. Cap Q,
fluxonium ind Q and quasiparticle channels are independent. Optional SQUID
charge/flux line losses require explicit impedance/mutual data. No fluxonium
inductive/qp formula is applied to the SQUID. Active exactly degenerate secular
transitions raise; near-degeneracy validity remains a model assumption.

`numeric_model` returns an independent `NumericModel` with continuous H(flux(t))
and piecewise instantaneous baths in the same fixed coordinates. Bath updates use
segment-start flux. Explicit flux callback events, including delayed pulse
support and envelope/phase changes, also refresh the baths and partition the
continuous Hamiltonian. The requested segment length rounds to the nearest 2 ns,
with a minimum of one tick. Evolution unions these model events with source
boundaries. It owns integration and never resets the density at a bath update.
`fixed_numeric_model` supplies the same Hamiltonian and full baths at an explicit
constant reduced flux, without callbacks or bath-update boundaries. It is not
valid for a driven flux source or passive tail. `FluxDynamics` keeps the callback's actual event coordinates and optional
stable dependency identity together for persistent direct records. Numeric matrices,
basis and actual controls still participate in their separate exact key.

`collapse_operators` combines multilevel relaxation with CP flux/residual-photon
pure dephasing. Slopes/pulls use rad/us in the pure-rate formulas, then rates
convert to 1/ns. No second T1/2 is added. Flux absolute-distance and photon
Lorentzian Gram kernels define diagonal Lindblad operators. This models an
exponential echo characteristic rate, not complete 1/f noise or Floquet baths.

`readout_frequencies_ghz` sums parent virtual transitions over the entire retained
local product space, beyond the M evolution states. It includes counter-rotating
terms and provides a response for every retained leakage/coupler state. Native
states outside that local parent remain truncated by user choice. No zero-pull
sink or runtime dispersive/convergence policy is inserted.

`secular_model(joint, level, sources, resonators=...)` supplies instantaneous
energy-basis energies, microwave quadratures, directional rates and pure rates.
Levels are integer F1/C/F2 offsets from the round's absolute DC flux, with
1e-3 Phi0 spacing. `model_source` wraps this builder in evolution's thread-safe
cache with DC Gibbs density and an exact `flux_expansion`. Model identities
include absolute DC, level, circuits, temperature, resonators and drive operators,
so an engine can share its prepare cache across rounds without aliasing DC points.
The collapse-operator and secular representations share the pure-rate calculation.

`FluxTrack` carries one source slot across all template points in ns/GHz/rad.
`schedule` filters source signals before loop mixing, then quantizes each DAC
interval's midpoint and merges equal levels. `flux_waveform` retains unsnapped
flux on DAC nodes for linear interpolation. Both use an absolute tail bound of
5e-4 Phi0 per loop, summed over absolute pulse contributions; waveform appends a
terminal DC sample. Filtering integrates sampled linear complex envelopes and
carriers analytically across the point arrays, without per-point quadrature.
`readout_model(joint, resonators)` supplies the existing ADC model at DC, with a
zero reference frequency and neutral receiver for composition to calibrate.

This package takes numeric GHz/ns/K values only. `baths` owns directional spectra,
`dephasing` owns pure-rate formulas, `lines` owns causal scalar delay/lowpass,
`evolution` owns density propagation and `adc` owns shared-line conditional fields
and sampling. Resources reads YAML and experiment composes these capabilities.
