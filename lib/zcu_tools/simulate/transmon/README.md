# SQUID transmon simulation

**Last updated:** 2026-10-27

**讀者：** framework-dev。SQUID transmon predictor 的輸入、查詢與失敗契約。

`TransmonParams` holds EJmax and EC in GHz and junction asymmetry d in [0, 1].
None means unknown. `missing()` lists unknown fields. `build_predictor()` rejects
incomplete or invalid inputs with `simulate.predictor.PredictorInputError`, a
ValueError leaf, and builds a fresh model. Numeric parameters contain no device
coordinates. Callers convert device values through `analysis.flux` before querying.
The model never reads or changes a component or a file.

## Spectrum and queries

`TransmonPredictor((EJmax, EC, d))` uses `scqubits.TunableTransmon` at ng=0,
with charge states n=-30..30. It diagonalizes H=4EC n²−EJ(φ) cos(phase), where
EJ(φ)=EJmax sqrt(cos²(πφ)+d² sin²(πφ)). It does not use a large-EJ approximation.
For d=0 at half flux, the spectrum is the free rotor E=4EC n², including the
±n degeneracies. d=1 gives a flux-independent spectrum.

Scalar `predict_freq(cur_flux)` and batch `predict_frequencies_mhz(fluxs, transitions)`
take reduced flux in Phi0 and return MHz. The batch returns the input flux grid
and frequency rows. Rows follow transitions, columns preserve input order and
duplicates, including unwrapped flux. Increasing or equal nonnegative integer
levels through level 20 are supported. Matrix curves are not provided.
All queries leave inputs and parameters unchanged. Invalid queries raise
PredictorInputError. Nonfinite computed spectra and numerical solver failures
propagate without translation or fallback.

## Calibration

`calculate_offset_correction(cur_flux, cur_freq, transition)` takes current
reduced flux including the caller's offset, measured MHz, and increasing integer
levels. It returns immutable `CalibrationResult(offset_correction, frequency_mhz, clamped)`.
Add offset_correction, in Phi0, to the caller's current offset. The query installs
nothing. Invalid inputs raise PredictorInputError; numerical failures propagate.

Both transmon and fluxonium invert their own spectra. They select the smallest
absolute correction across all roots and periodic/mirror equivalents relative
to current flux. Unreachable targets return the closest attainable frequency
and clamped=true. Equally close frequencies select the smallest absolute correction.
Reachable targets return clamped=false. A constant spectrum selects zero correction.

The shared solver samples 401 points across one half period to discover branches,
then refines local extrema and brackets roots using exact spectrum evaluations.
It includes interior extrema and tangential roots, without assuming a monotone
half period. Features narrower than the branch-discovery grid can require higher
numerical resolution. Frequency equality uses an absolute tolerance of 1e-7 MHz;
root refinement uses 1e-13 flux quanta.
