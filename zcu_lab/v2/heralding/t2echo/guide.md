# Heralded T2 Echo

**Readers:** operator, agent. Run heralded t2 echo without a classifier and review self-calibrated state-change fits and writeback.

## Before Run

Select calibrated π and π/2 pulses from `$qubit.module.x180` and `$qubit.module.x90`, plus single-tone `$resonator.module.readout`. Mux is not supported. Reset is optional. Delay defaults to 1.5 × captured T2e and Relax delay to 5 × captured T1; manually fill missing calibration. Templates still require channels, gains, frequencies and timing.

## Run and analyze

The sequence is optional reset → herald → π/2 → half delay → π → half delay → phase-shifted π/2 → signal. Delay is total free-evolution time in us; each half is hardware-quantized. Detune ratio/step MHz ramps only the final phase. Zero ratio disables that ramp; nonzero ratio with zero step fails. Early-stop SNR uses C with its own calibration cache. Initial or current calibration failure cannot stop the round.

Select `n_states`, `fit_method` and `fit_phase`. Full calibration precedes finite delay/C filtering and removal of the first remaining point. Method decay fits an exponential; fringe fits decaying fringes. Free fringe phase is optional; otherwise phase is fixed to 0°. Source cfg detune, or run-only realized detune without cfg, must be finite for fringe fitting. Zero detune forces decay. Unknown methods and insufficient data fail rather than selecting another model.

The figure contains only C and the fit with π̂ in the legend. Summary returns `t2e`, `t2e_err` in us, `detune`, `detune_err` in MHz, `pi_hat`, normalized `singular_values` and `fit_quality["fit"]`. Decay-only detune and error are zero. No pointwise arrays are returned. Identification errors propagate.

## Review writeback

Ordinary Echo targets and rules remain, marked Heralded: `$qubit.t2e` and an eligible captured-bias record. Default selection requires positive finite T2e, relative error ≤0.5, R² ≥0.9 and normalized RMS ≤0.1. Missing or unsuitable diagnostics produce review notes. Inspect before Apply; analysis does not write calibration. A finite-temperature Markov bath model does not certify hardware coherence or low-frequency-noise refocusing.

## Readout and calibration

Herald and signal use the same single-tone pulse readout. No classifier or singleshot calibration is required. Acquisition saves the outer-product sums of ψ(h) and ψ(s), where ψ(z) = (1, I, Q, I², IQ, Q²). Reps counts attempted shots at each point, and Rounds adds their sums. Heralding adds no feedback or active reset.

Analysis selects `n_states`, default 2, or 3. Second-order IQ can identify at most three states. Three collinear centers, insufficient state population or a failed lift make analysis fail. Readout tails and transitions during integration can bias calibration. Herald and signal must share readout centers and isotropic Gaussian width.

The fitted observable is C(x) = 1 − tr J(x), the probability that the state changes between herald and signal. K[i,j] = P(i → j) is a transfer diagnostic, not a fitted curve or gate fidelity. The legend and summary report `pi_hat`, the population entering the intermediate operation after herald. Decay during herald can lower its excited population relative to the original thermal state. States are ordered by decreasing population, not assigned g/e/f labels from IQ position.

## Live display

Before the first successful two-state calibration, the measurement plot says Calibrating and has no C points. Later updates warm-start from the previous successful calibration. An identification failure leaves the last plot unchanged and acquisition continues. Analyze always recalibrates the full acquired data from scratch using the selected state count. It never reuses the live calibration.

## Save and load

Native Save stores float64 `sums` with shape (N, 6, 6), the grid, acquisition cfg and metadata. The last two array axes are `herald_features` and `signal_features`, both ordered (1, I, Q, I², IQ, Q²). `sums[:, 0, 0]` is the shot count. Unmeasured points stay NaN. Time coordinates use seconds in the native file and us in Result; gain coordinates retain native units.

Old classified-herald or three-component signal files fail the existing axis-label validation at Load. There is no compatibility path. A missing `sums` variable also fails Load. Load restores the saved record, not the current Form. Analyze again after Load. Check each output's save status; native saving does not imply every export succeeded.

## MCP recipe

`heralding_t2echo` selects sources, applies inputs and overrides, checks readiness, runs, analyzes, asks about writeback and saves completed data even if analysis fails. `points` maps to Expts. Existing analysis inputs are fit_method and fit_phase; set `n_states` through analysis overrides. Omitted inputs retain Form values; explicit null passes unchanged to the field owner and may be rejected.
