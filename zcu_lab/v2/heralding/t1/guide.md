# Heralded T1

**Readers:** operator, agent. Run heralded t1 without a classifier and review self-calibrated state-change fits and writeback.

## Before Run

Select a calibrated π pulse from `$qubit.module.x180` and a single-tone readout from `$resonator.module.readout`. Mux is not supported. Set Delay Start/Stop/Expts in us. Linear sampling uses a uniform FPGA sweep. Decay sampling uses the shared T1 grid and rejects colliding hardware-quantized points. Set Reps, Rounds and Relax delay. Simulation readout gain remains subject to its dispersive-validity guard.

## Run and analyze

The sequence is herald → π pulse → swept delay → signal. Select `n_states`, `skip` and `dual_exp`. Calibration uses all measured points. Skip then removes leading points on the original delay grid, and fitting omits nonfinite delay/C pairs. The default model is a + b·exp(−t/T1); `dual_exp=True` uses two decay terms.

The fit plot contains only C and the combined fit, with π̂ in the legend. `t1` and `t1_err` use us. `t1b` and `t1b_err` are None for a single decay and use us for two decays. Summary also returns `pi_hat`, normalized `singular_values` and `fit_quality["fit"]`. No pointwise arrays are returned. Identification and fitting errors propagate.

## Review writeback

Apply writes `$qubit.t1` only after explicit operator confirmation. Its destination and proposal rules are unchanged.

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

`heralding_t1` selects sources, applies inputs and overrides, checks readiness, runs, analyzes, asks about writeback and saves completed data even if analysis fails. `points` maps to Expts. Existing analysis inputs are dual_exp and skip; set `n_states` through analysis overrides. Omitted inputs retain Form values; explicit null passes unchanged to the field owner and may be rejected.
