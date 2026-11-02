# Heralded amplitude Rabi

**Readers:** operator, agent. Run heralded amplitude rabi without a classifier and review self-calibrated state-change fits and writeback.

## Before Run

Bind qubit and resonator. Choose the fixed drive frequency, waveform and duration, readout generator/ADC settings and timing. A prior pi amplitude is not required. Set Gain Start/Stop and Expts to cover oscillations within your approved drive limits. Review Reps, Rounds and Relax delay; suggested values are not hardware-safety approval.

## Run and analyze

The sequence is optional reset → herald → optional initialization → gain-swept drive → signal. Both readout instances use the same cfg. Open Analysis and select `n_states` and `skip`. Calibration uses every measured point, then skip removes leading points on the original gain grid. Nonfinite gain/C pairs are omitted from fitting.

The fit is a + b·cos, with no decay envelope. C nearest zero gain selects fixed phase 0° when above the trace midpoint, otherwise 180°. The fit plot contains only C and its fit, with π̂ in the legend. The summary returns `pi_amp`, `pi_amp_err`, `pi2_amp` and `pi2_amp_err` in native gain units, plus `pi_hat`, normalized `singular_values` and `fit_quality["fit"]`. It contains no pointwise J, K or C arrays. Negative skip, identification failure and insufficient fit data raise errors.

## Review writeback

Writeback retains ordinary amplitude Rabi's five destinations, marked Heralded. It proposes pulse copies at `$qubit.pulses.pi_amp` and `$qubit.pulses.pi2_amp`, their `$qubit.module.x180` and `$qubit.module.x90` links, and `$qubit.ext.pi_product` in gain·us. Captured sources and the scalar source duration are required. Without saved cfg there are no proposals. Review every item before Apply; Analyze does not write calibration.

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

`heralding_amp_rabi` selects sources, applies inputs and overrides, checks readiness, runs, analyzes, asks about writeback and saves completed data even if analysis fails. `points` maps to Expts. Existing analysis inputs are skip; set `n_states` through analysis overrides. Omitted inputs retain Form values; explicit null passes unchanged to the field owner and may be rejected.
