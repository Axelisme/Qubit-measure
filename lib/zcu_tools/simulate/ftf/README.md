# Fixed-coordinate FTF numerics

**Last updated:** 2026-10-28 (UTC), flux-pulse routing and fixed-projection limits

**讀者：** framework-dev、agent。固定座標 FTF 數值模型、路由與快取契約。

`JointParameters` supplies F1/C/F2 numeric circuits and signed full-charge
couplings in GHz. `JointModel` selects fixed local levels at DC, forms their
product space with all g1c/g2c/g12 terms, then selects the lowest M joint states.
Dynamic Hamiltonians stay in this fixed basis. Each joint retains up to eight
exact-input eigensystems and bath-rate results as read-only arrays, within its own
fixed projection. Native fluxonium operators use a separate eight-entry cache
keyed by exact EC, EL and oscillator cutoff; projected operators never enter that
cross-joint cache. Neither memo changes the numerical operation order.
Native n² is formed before local
and joint projection; squared truncated n is not a substitute.

`charge_operator(subsystem)` returns full dimensionless n in the retained basis.
`flux_operator(subsystem)` returns the analytic derivative of this model's H/h
with respect to reduced flux f = Phi/Phi0, evaluated at the round's DC point,
in GHz/Phi0. Both use subsystem 0=F1, 1=coupler, 2=F2. The flux derivative is
projected through the same fixed local and retained joint bases as H, without
rediagonalization or a gauge change. In the oscillator coordinates used here,
fluxonium has H/h = kinetic - EJ*cos(phi + 2*pi*f), so its derivative is
2*pi*EJ*sin(phi + 2*pi*f), not a bare phase operator. The fixed-theta SQUID
derivative is pi*EJmax*[sin(pi*f)*cos(theta) + d*cos(pi*f)*sin(theta)].
Drive composition multiplies these by GHz/gain charge coefficients or Phi0/gain
flux coefficients. These drive quadratures are a fixed-DC small-signal
linearization. Separate `FluxTrack` inputs change H(Phi(t)) through the exact
`flux_expansion` within the same retained DC projection; they do not use this
derivative approximation.

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

`numeric_model` returns an independent `NumericModel` containing the static DC
Hamiltonian, supplied drive sources and event metadata, without collapse
operators. Regular event spacing rounds to the nearest 2 ns, with a minimum of
one tick. Quantized `FluxDynamics` events in the requested window join those
ticks and zero. Its optional identity remains caller metadata.
`fixed_numeric_model` supplies a static Hamiltonian and full baths at an explicit
constant reduced flux, without event metadata. It is not valid for a driven
flux source or passive tail. Neither record performs evolution. Engine consumes
`ModelSource` from `model_source`, not `NumericModel`.

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
so retained source models and routing facts cannot alias different DC points.
The collapse-operator and secular representations share the pure-rate calculation.

`FluxTrack` carries one source slot across all template points in ns/GHz/rad.
`schedule` filters source signals before loop mixing, then quantizes each DAC
interval's midpoint and merges equal levels. `flux_waveform` retains unsnapped
flux on DAC nodes for linear interpolation. Each line implements the public
`FluxLine` step response as a base pole of weight `1-sum(A_k)` plus distortion
poles of signed weights `A_k`. A zero base tau gives instantaneous feedthrough.
Filtering integrates sampled linear complex envelopes and carriers analytically
across the point arrays, without per-point quadrature. Empty distortion terms
retain the original single-pole calculation.

Both functions use an absolute tail bound of 5e-4 Phi0 per loop. Each pulse's
maximum absolute envelope, gain and loop coefficient scale the sum of absolute
pole weights. The slowest pole bounds their decay after delayed pulse end.
The error budget is divided among source slots, so opposite signed poles or
sources cannot cancel the bound. The waveform appends a terminal DC sample only
after that bound is below tolerance. `schedule` rounds midpoint offsets from DC
to the nearest 1e-3 Phi0 level, ties even, and merges equal adjacent levels.
Offsets smaller than 5e-4 Phi0 snap to DC. These functions do not propagate a
density or certify convergence of the fixed-DC projected subspace.
`readout_model(joint, resonators)` supplies the existing ADC model at DC, with a
zero reference frequency and neutral receiver for composition to calibrate.

## Flux-pulse evolution and limits

The experiment layer binds optional F1/C/F2 flux lines to `FluxTrack` inputs.
Ideal constant baseband pulses with zero tau and no distortion omit the
continuous waveform. With no microwave drive, the engine uses closed-form `t4`
segments and sudden basis changes at flux steps. Filtered, sampled-varying or
nonzero-carrier flux supplies a continuous waveform. No-drive motion rows use
`adiabatic_map`; driven continuous flux uses `flux_dop853`. A truly constant
no-drive park row uses `t4`, even within a sequence that has continuous ramps.
There is no separate sudden/adiabatic setting.

`adiabatic_map` preserves instantaneous ordered eigenstate populations, even
when the quantized row bath has nonzero transition rates. Those rates only
contribute coherence decay on this route. Neglecting population relaxation on
a ramp is reasonable only when the ramp is much shorter than T1. Ramps near T1
underestimate relaxation. A T1 reset therefore needs an actual constant park
row routed to `t4`; a filtered tail or an entirely `adiabatic_map` sequence does
not establish reset. The caller must also allow return-to-DC settling before
terminal readout, whose model remains fixed at DC.

The exact flux expansion is exact only inside the retained fixed-DC subspace.
It does not restore states discarded by local or joint projection. Users must
check native cutoffs, local levels and retained joint levels for their sequence.
Legal density matrices, successful evolution and small local-level differences
do not certify retained-level convergence. Adiabatic quadrature convergence
also does not certify slow motion or a nondegenerate, noncrossing spectrum.
The [experiment reset example](../../experiment/ftf/README.md#reset-interpretation)
checks small-model mapping and constant-hold relaxation, not device fidelity or
large-excursion convergence.

## Cavity drive composition

`CavityDrive.advance` averages one span's photons, Stark shifts and driven
coherence decay, then returns its main branch's final field. The main branch
has the largest span-start population; exact ties select the lowest DC energy.
Every conditional branch begins with the same inherited field at that span's
absolute start time. ADC owns propagation and second-moment integration. This
single-branch memory is semiclassical and does not describe quantum cavity-state
entanglement. Experiment composes it with evolution's continued DC densities.
An optional caller-owned `CavityAverageCache` supplies exact flat span averages.
`CavityDrive` only selects the ADC entry point and forwards physical inputs; it
owns neither cache keys nor storage policy. Final fields and branch selection
remain uncached.

## Readout and Floquet paths

FTF passes readout populations and the DC readout model to `simulate.adc`.
For a single readout pulse with `envelope=None`, the batch response uses the
exact matrix exponential in the carrier frame, even when the carrier differs
from the model reference. Pulses with envelopes and multi-tone readout use the
general DOP853 moment ODE on this acquisition path. Flat mux multi-tone pulses
also use this ODE.

Two constant microwave tones can use evolution's Floquet k=0 path when one is a
weak probe and the Floquet guards pass. The strong tone defines the effective
Hamiltonian and dissipator. The probe uses modal second-order Dyson for arbitrary
input densities, with periodic T6 propagation when its guard fails. Segments
that fail the Floquet guards use the summed-drive DOP853 path. This approximation
does not add a physical drive-dressed bath to the FTF model.

This package takes numeric GHz/ns/K values only. `baths` owns directional spectra,
`dephasing` owns pure-rate formulas, `lines` owns causal scalar delay/lowpass,
`evolution` owns density propagation and `adc` owns shared-line conditional fields
and sampling. Resources reads YAML and experiment composes these capabilities.
