# Heralded length Rabi

**Readers:** operator, agent. Run heralded length rabi without a classifier and review self-calibrated state-change fits and writeback.

## Before Run

Bind qubit and resonator. Choose the fixed drive frequency/gain, waveform, readout generator/ADC settings and timing. A prior pi duration is not required. Set Duration Start/Stop in us, Expts, Reps, Rounds and Relax delay. Select optional reset and initialization as needed. Const/flat-top use an FPGA sweep. Other editable waveforms use a round-major host scan. Native arb duration sweeps are rejected.

## Run and analyze

The sequence is optional reset → herald → optional initialization → length-swept drive → signal. Host segments and completed rounds add moment sums. Early-stop SNR uses C after a complete curve, with an independent calibration cache. Before calibration succeeds, or if that check's calibration fails, it cannot stop the round. Acquire retry retains the ordinary policy.

Select `n_states`, `decay` and `fit_phase`. After full calibration, the fit omits nonfinite length/C pairs. `decay=False` fits a + b·cos; True adds a decaying envelope. `fit_phase=False` fixes phase at 0°; True fits it. The figure contains only C and its fit with π̂ in the legend. Summary fields `pi_len`, `pi_len_err`, `pi2_len`, `pi2_len_err` use us; `rabi_f` and `rabi_f_err` use MHz. It also returns `pi_hat`, normalized `singular_values` and `fit_quality["fit"]`, not pointwise arrays. Identification and fitting failures propagate.

## Review writeback

Ordinary length Rabi's targets remain, marked Heralded: `$qubit.ext.rabi_freq`, `$qubit.pulses.pi_len`, `$qubit.pulses.pi2_len`, `$qubit.module.x180`, `$qubit.module.x90` and `$qubit.ext.pi_product`. The amplitude-calibration pulses remain unchanged. Without cfg only the frequency proposal is available. Captured sources and scalar gain are required for pulse/product proposals. Review before Apply Selected.

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

`heralding_len_rabi` selects sources, applies inputs and overrides, checks readiness, runs, analyzes, asks about writeback and saves completed data even if analysis fails. `points` maps to Expts. Analysis inputs are `decay` and `fit_phase`. Omitted Form inputs and analysis controls retain their owner values. Explicit null passes unchanged to the field or analysis owner and may be rejected.

The recipe has no `n_states` input, and Form overrides cannot set analysis options. To change the state count, edit `n_states` in the completed tab's Analysis settings and rerun Analyze. The Analysis default is 2; second-order IQ supports at most 3 states.
