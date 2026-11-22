# T1 from paired readout moments

**Readers:** operator, agent. Run `twotone/t1_cov`, assess the state-change fit, and review T1 writeback.

## Prepare the sweep

Select a calibrated π pulse from `$qubit.module.x180` and a single-tone pulse readout from `$resonator.module.readout`. Both reads use that readout. You do not need a classifier, and mux readout is not supported.

Select an optional pulse reset from `$qubit.module.reset` to prepare a repeatable initial distribution, or leave Reset empty. Each shot plays reset if selected, `pre_readout`, the π pulse, the swept delay, then `post_readout`. The experiment does not select or discard shots based on their IQ.

Set the delay window in µs and the number of points. Linear sampling uses a uniform FPGA sweep. Decay sampling puts more points near the start and rejects points that coincide after hardware quantization. Reps sets the number of shots per point per round. Rounds adds the sums from completed rounds. Relax delay sets the wait after each shot in µs.

## Read the live plot and fit

The live plot says Calibrating until a two-state calibration succeeds. Later updates reuse the last successful calibration as a starting point. A failed update leaves the current plot on screen and acquisition continues.

Analyze recalibrates from all measured points. Choose `n_states=2` or `3`, with 2 as the default. Three states require noncollinear IQ centers. Both reads must share IQ centers and approximately isotropic Gaussian noise of the same width. Weakly populated states, readout tails, and transitions during integration can make calibration fail or bias the result.

The fit uses $C = 1 - \mathrm{tr}(J)$, the probability that the state differs between the two reads. $J$ is the joint probability of the two states. The conditional transfer matrix is a calibration diagnostic, not the fitted observable or a gate fidelity. `skip` omits leading points from the fit, and the fit also drops nonfinite delay and C pairs. Select `dual_exp` to fit two decay terms instead of one.

The fit plot shows C and the fitted curve. The legend reports `pi_hat`, the populations after `pre_readout`, ordered by population rather than assigned g/e/f labels. Decay during the first read can lower the excited population. The summary reports T1, its error in µs, normalized singular values, and fit diagnostics. The second lifetime and its error are empty for a single-exponential fit.

## Review writeback and save

Check the fit and its uncertainty before accepting the `$qubit.t1` proposal. Nothing is written to the qubit unless you accept it.

Native Save writes `twotone/ge/t1_cov` with the cfg, metadata, and float64 `sums` of shape `(N, 6, 6)`. The array axes are time, `pre_readout_features`, and `post_readout_features`. Each feature axis is ordered `(1, I, Q, I², IQ, Q²)`. Each entry sums a product of the two reads' features over the shots. `sums[:, 0, 0]` counts the shots per delay. Unmeasured points stay NaN. Time is stored in seconds and restored in µs.

Load accepts the new tag and restores the saved record, not the current Form. Older experiment tags are not accepted. Run Analyze again to change the fit options after Load.

## Run the recipe

The `t1_cov` recipe takes the shared readout and π pulse keys, an optional `use_reset` selection, and delay and acquisition controls. `points` sets Expts. Omitted inputs retain their current values, and null is passed to the field owner. The recipe applies `overrides` before checking readiness.

The recipe runs, analyzes, asks before writing T1, and saves completed acquisition data even if analysis fails. Set `skip` and `dual_exp` with recipe inputs. To change `n_states`, edit the tab's Analysis settings and run Analyze again.
