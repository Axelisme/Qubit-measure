# T1 from paired readout moments

**Readers:** operator, agent. Measure relaxation from the probability that the qubit state differs between two reads in the same shot.

## Sequence and key settings

Optional **Reset** → first readout → calibrated **Pi Pulse** → swept **Relaxation delay** → second readout → **Relax delay**. Use the same calibrated single-tone pulse **Readout** for both reads, not mux. No classifier is needed and no shots are selected or discarded by IQ.

Choose a delay window in µs that resolves the decay; decay sampling concentrates points at early times and rejects a grid with coincident applied delays. For Analyze, set **Number of states** to 2 or 3. Three states need noncollinear IQ centers. Both reads must share centers and approximately isotropic noise of similar width. Weak states or transitions during integration can bias calibration. **Skip leading points** omits early fit points; **Dual exponential** adds a second decay term.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

The live plot says **Calibrating** until the two-state calibration succeeds; a failed update keeps the previous plot and acquisition continues. Analyze recalibrates on all measured points. The plotted quantity is `C = 1 − tr(J)`, where J is the joint state probability inferred from paired IQ moments. C is not a classified shot count or gate fidelity. Finite-sample C can lie outside [0, 1]; check IQ separation and the shared-center assumption instead of clipping it. The plot does not use binomial shot SEM.

Analyze overlays the decay and reports T₁ and its fit standard error, not pointwise uncertainty. `pi_hat` lists populations inferred after the first read, ordered by population rather than labeled g/e/f. The fit's R² compares residual and measured variation; it does not validate the IQ calibration.

## Next step

Writeback proposes `$qubit.t1`. Check that the fit covers the decay and that its uncertainty supports the lifetime. Load accepts native paired-readout records, not older experiment tags.

For `t1_cov`, null disables **Reset**. Null **Readout** or **Pi Pulse** fails during selection, before overrides. Null sweep Stop/Expts, Reps or Rounds leaves a blank; required blanks left after overrides fail readiness before Run. Null **Dual exponential** or **Skip leading points** fails at Analyze. Set **Number of states** on the tab; the recipe does not override it.
