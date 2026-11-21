# Thermal-chain readout ADC window sweep

**Readers:** operator, agent. Run the no-drive ADC window sweep and read its heuristic score, QND and correlation-growth diagnostic.

**Last updated:** 2026-11-21

After changing a parameter source, use Reset to recapture initial values. Refresh updates sources and choices while preserving authored inputs. Reset can overwrite unsaved inputs.

## Shot and parameters

`twotone/ro_optimize/chain_length` sweeps readout ADC window in us.
Each shot reads one PulseReadout 16 times without a drive or reset.
After a readout pulse ends, the readout's `post_delay` gives cavity photons time to decay before the next readout starts.
The next pulse keeps its own pre-delay but may start before the previous ADC window closes. Set the timing so windows on the same ADC do not overlap.
The relax delay K follows each shot. Every shot after the first starts K after the previous shot.
Set `relax_delay` in us long enough to return to thermal equilibrium.
The Form seeds it from 5·T1. The first shot relies on the state before Run.
There is no additional wait inside the chain beyond the readout's own timing.

Readout uses the captured `$resonator.module.readout` slot, or a fresh `readout` template.
The template uses the resonator frequency, wiring and time of flight.
ADC window starts at 11 points from 0.2 to 2.2 µs. At each point, Run sets the ADC window to the sweep value and the pulse duration to `window + length_margin`. The Form hides the two lengths that Run sets.

**Length margin (µs)** is editable and must be finite and nonnegative. On Open or Reset it starts at the captured readout's `waveform.length − ro_length`. Without a captured readout it starts at the difference in the `readout` template. If either length cannot be resolved, enter the margin before Run. Refresh keeps the existing input.

Keep the trigger offset and pre-delay from a readout that covers the ADC window. Sweeping both lengths by the same amount preserves their timing at the ADC, apart from clock rounding. If you change the margin or the trigger, verify that the pulse reaches the ADC before it opens and still covers it when it closes. The sweep does not adjust the trigger offset or require a separate flight-time input.
Reps defaults to 2000 shots per setting per round. Rounds defaults to 320, giving 640 000 shots per setting.
Supply any other missing or invalid sources before Run.

This experiment needs thermal g/e contrast, not a calibrated π pulse.
The state after K is not the readout-driven steady state. A measured 60:40 readout-driven population does not specify the population before the first read.
Traditional `ro_optimize` experiments keep their drive-based preparation and analysis.

## Score and selection

Analysis projects early covariances onto their main IQ axis.
It estimates ρ as the first-pair covariance divided by R1 variance, and QND from the decay of the first four R1 cross covariances.
G is a restricted two-state, 15-lag information amplification proxy.
It averages the model information ratio over determinant 0..1 with a fixed 24-point quadrature.
It targets a symmetric driven ...DRDRDR... model. It does not assert that this model and K R^16 have the same physical information.
An unequal initial thermal population reduces the effective contrast. The inferred contrast is not a calibrated g/e SNR.

M measures within-chain correlation growth.
Analysis centres each read separately, symmetrises each adjacent covariance matrix, and weights pairs r=1..15 by (r−8)/20.
It takes the largest eigenvalue of the summed matrix and divides by R1 variance.
M is a diagnostic, not an identified leakage probability or physical MIST amount.
Physical MIST means leaving g/e. Pure two-state repopulation can also increase M and incur a penalty.

The combined score is ρ²·G·[|ρ|/(|ρ|+max(M,0))]².
At ρ=0 the score and its error scale are zero.
Only the maximum score selects the best setting. Exact ties keep acquisition order.
Errors do not affect ranking, and no second threshold excludes settings.
If the best ADC window is at either scan boundary, extend that side and repeat before using it.

## Live and analysis plots

Live `measurement` and offline `metric` figures have the same three panels:

- Heuristic score with model-conditional approximate error bars.
- QND with model-conditional approximate error bars. The axis stays at 0..1; reports retain unmodified QND, even outside that range.
- M alongside |ρ|. Large M remains visible on this diagnostic axis.

Red dashed lines mark the maximum-score setting in every panel.
The score axis uses score values alone, so large errors do not flatten the curve.
Error bars can extend beyond that axis and be cut off.
Acquisition calculates scores after each completed round, then the plot thread draws them.
Stop keeps settings with a completed round. A stop before any completed round raises an error.

Analysis has no options and can use saved sums without cfg or hardware.
The summary is `best`, a SettingScore containing `setting` in us, `score`, `score_error`, `qnd`, `qnd_error`, `correlation`, `amplification` and `mist`.
The API name `mist` holds M, with the diagnostic meaning above.
There is no fit, no-signal classification or third-state population estimate.
Invalid projected variance or zero early covariance norm raises ValueError rather than inventing a QND value.

## Errors and known limits

The error scales are model-conditional approximations, not general one-sigma errors or calibrated confidence intervals.
Score propagation holds M and the main axis fixed and propagates only ρ and QND.
It ignores their shared covariance, M uncertainty, main-axis uncertainty and selection of the maximum.
G uses clipped QND and effective contrast to remain finite.
At and beyond clipping, propagation uses the model slope at the clipped coordinates, not the true derivative of the clipped output.

Real K R^16 data have not yet validated this score.
A rare, distant thermal third state or correlated read noise can raise the score without raising M.
A third state overlapping g or e in IQ may hide leakage. Anisotropic noise can select the wrong main axis.
Pure two-state drift can increase M when the initial thermal state differs from the driven steady state. The penalty does not separate this from leakage.
Fixed pair weights retain that false-penalty limitation.
For an increase confined to the last pair, the independent isotropic low-SNR model predicts about 59% of the endpoint-difference detection SNR. This is a model result, not a device measurement.
Finite-shot noise can give a positive score even without useful g/e contrast.
Vanishing g/e contrast alone does not force score to zero if a third state remains visible.
ρ uses R1 variance rather than both endpoints' standard deviations, so arbitrary nonstationary moments have no promised global score bound.
Approximate errors do not guarantee selection of the true best setting.

## Saved data and MCP recipe

The record is ThermalChainSums with the sweep axis, shot count and 107 float64 sums per setting.
It keeps I/Q sums of R1..R16, R1 I²/IQ/Q², all four cross components of all 15 adjacent pairs, and the R1,R3 through R1,R5 pairs.
No weighted M aggregate replaces those pairs.
Native save and restore preserve the moments for reanalysis of score, M and best setting.
The tag is `twotone/ro_optimize/chain_length` and cfg schema is `6.0`. An older run can still load its saved moments for analysis, but its cfg is not restored.

`ro_optimize_chain_length` accepts optional `length_start_us` and `length_stop_us` in us, `points`, `relax_delay`, `reps` and `rounds`.
`points` maps to sweep Expts. Omission keeps the Form value; null passes to the field owner.
Overrides apply last, then check_ready rejects missing or invalid inputs before Run.
The recipe runs, analyzes and saves. It proposes no writeback.
It reports `length_us`, `score`, `score_error`, `qnd`, `qnd_error`, `correlation`, `amplification` and `mist`.
To use the chosen ADC window in the readout module, set the pulse duration to that window plus the margin used in the sweep.
