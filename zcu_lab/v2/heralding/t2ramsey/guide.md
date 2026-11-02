# Heralded T2 Ramsey

**Readers:** operator, agent. Run heralded t2 ramsey without a classifier and review self-calibrated state-change fits and writeback.

## Before Run

Select a calibrated π/2 pulse from `$qubit.module.x90` and a single-tone readout from `$resonator.module.readout`. Mux is not supported. Reset is optional. Open/Reset captures sources; Refresh does not overwrite edited inputs. Delay starts at 0 and ends at 1.5 × captured T2r, with 101 points. Relax delay starts at 5 × captured T1, or 150 us if T1 is absent. These editable starting values are not hardware-safety approval.

## Run and analyze

The sequence is optional reset → herald → π/2 → delay → phase-shifted π/2 → signal. Detune ratio, default 0.05 fringe cycles per delay step, becomes ratio/step MHz. It ramps only the final phase, not drive frequency. Nonzero ratio with zero step fails before acquisition. Reps and Rounds count attempted shots and software rounds. Early-stop SNR uses C with its own calibration cache. Initial or current calibration failure cannot stop the round.

Select `n_states`, `fit_fringe` and `fit_phase`. After full calibration omit nonfinite delay/C pairs. `fit_fringe=True` fits decaying fringes; False fits a pure exponential. Free phase is the default; disabling it fixes 0° or 180° according to C nearest zero delay. Zero detune does not automatically disable fringe fitting.

The figure contains only C and the fit with π̂ in the legend. Summary returns `t2r`, `t2r_err` in us, `detune`, `detune_err` in MHz, `fit_fringe`, `pi_hat`, normalized `singular_values` and `fit_quality["fit"]`. Decay-only detune and its error are zero. No pointwise arrays are returned. Identification and fitting errors propagate.

## Review writeback

Proposals retain ordinary Ramsey's rules and Heralded descriptions. `$qubit.t2r` receives lifetime. A fringe fit with saved cfg and run-only realized detune can propose `$qubit.freq` = source pulse frequency + realized detune − fitted detune. Captured qubit and device snapshot can also produce a captured-bias t2r record. Inspect proposals before Apply. Frequency changes do not update pulse-library frequencies.

Native Load does not restore run-only realized detune, so loaded data do not propose frequency correction. Lifetime and eligible bias-record proposals remain available.

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

`heralding_t2ramsey` selects sources, applies inputs and overrides, checks readiness, runs, analyzes, asks about writeback and saves completed data even if analysis fails. `points` maps to Expts. Existing analysis inputs are fit_fringe and fit_phase; set `n_states` through analysis overrides. Omitted inputs retain Form values; explicit null passes unchanged to the field owner and may be rejected.
