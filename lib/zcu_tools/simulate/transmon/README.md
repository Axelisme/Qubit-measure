# SQUID transmon simulation

`TransmonParams` holds EJmax and EC in GHz, junction asymmetry d in [0, 1],
and flux_half, flux_period, flux_bias in native device units. None means unknown.
`missing()` lists unknown fields. `build_predictor()` rejects incomplete or
invalid inputs and builds a fresh model. It never reads or changes a component.

## Spectrum and queries

`TransmonPredictor` uses `scqubits.TunableTransmon` at ng=0, with charge states
n=-30..30. It diagonalizes H=4EC n²−EJ(Φ) cos(φ), where
EJ(Φ)=EJmax sqrt(cos²(πΦ)+d² sin²(πΦ)). It does not use a large-EJ approximation.
For d=0 at half flux, the spectrum is the free rotor E=4EC n², including the
±n degeneracies. d=1 gives a flux-independent spectrum.

The affine is Φ=(value+flux_bias−flux_half)/flux_period+0.5. Period can have either
sign but cannot be zero. Scalar `predict_freq` and batch
`predict_frequencies_mhz` return MHz. Batch frequency rows follow the requested
transitions, columns follow the supplied device values. Increasing or equal
nonnegative integer levels through level 20 are supported. Matrix curves are
not provided. All queries leave inputs and model parameters unchanged.

## Calibration

`calculate_bias(value, frequency_mhz, transition)` returns the immutable
`simulate.predictor.CalibrationResult`, with flux_bias, actual frequency_mhz and
clamped. It requires increasing integer levels. Invalid input raises ValueError;
numerical solver failures propagate. A successful query never installs the bias.

Both transmon and fluxonium invert their own spectra. They select the smallest
absolute bias across all roots and periodic/mirror equivalents, independent of
the current bias. Unreachable targets return the closest attainable frequency
and clamped=true. Equally close frequencies select the smallest absolute bias.
Reachable targets return clamped=false. A constant spectrum selects zero bias.

The shared solver samples 401 points across one half period to discover branches,
then refines local extrema and brackets roots using exact spectrum evaluations.
It includes interior extrema and tangential roots, without assuming a monotone
half period. As with any finite branch-discovery grid, features narrower than the
grid spacing can require higher numerical resolution. Frequency equality uses
an absolute tolerance of 1e-7 MHz; root refinement uses 1e-13 flux quanta.

Frontends consume this result without recalculating the fit. The dialog changes
only its own bias/cache and displays actual MHz and clamped. MCP returns the
same facts without writing the component or dialog.
