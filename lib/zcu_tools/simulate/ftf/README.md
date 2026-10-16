# Fixed-coordinate FTF numerics

**Last updated:** 2026-10-16

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
segment-start flux. The requested segment length rounds to the nearest 2 ns,
with a minimum of one tick. Evolution unions these model events with source
boundaries. It owns integration and never resets the density at a bath update.
`fixed_numeric_model` supplies the same Hamiltonian and full baths at an explicit
constant reduced flux, without callbacks or bath-update boundaries. It is not
valid for a driven flux source or passive tail. Dynamic callback models can carry
a stable dependency identity for persistent direct records. Numeric matrices,
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

This package takes numeric GHz/ns/K values only. `baths` owns directional spectra,
`dephasing` owns pure-rate formulas, `lines` owns causal scalar delay/lowpass,
`evolution` owns density propagation and `adc` owns shared-line conditional fields
and sampling. Resources reads YAML and experiment composes these capabilities.
