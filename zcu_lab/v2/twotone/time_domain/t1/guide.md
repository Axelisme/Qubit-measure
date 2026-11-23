**Readers:** operator, agent. Measure T₁, the energy relaxation time after a π excitation.

## Sequence and key settings

Optional **Reset** → calibrated **Pi Pulse** → swept **Relaxation delay** → calibrated **Readout** → **Relax delay** between shots. Check the π pulse and readout timing before Run.

- The sweep starts at `0…5 × qubit.t1`; the inter-shot **Relax delay** also starts at `5 × qubit.t1`. With no T₁, both are based on 30 µs, giving 150 µs for each. An invalid stored T₁ leaves required values blank. Aim to see the signal reach baseline.
- Use decay sampling to put more points early in the same window. Use **Dual exponential** only if the trace has a clear second timescale. **Skip leading points** removes early points from the fit, not the acquisition.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Both plots show the principal-axis projection of mean IQ (a.u.), not population, versus relaxation delay (µs). The live title counts retained shots per point. One bar is one shot SEM: retained-shot sample SD along that axis divided by √(retained shot count). It is not shot spread, round-to-round scatter or fit error. Points without sufficient shot statistics have no bar; mean-only files still load. Bars do not weight the fit.

Analyze overlays the decay curve and reports T₁ with fit standard error; a dual fit adds a second timescale. R² compares the residual with the variation of the data, not an acceptance cutoff. If the trace does not flatten, extend the delay window and keep enough points early in the decay.

## Next step

Writeback proposes `qubit.t1` and its fit error, plus an independent T₁ sample when Run bias evidence is available. It does not change module slots. Review the curve and uncertainty before adopting it.

For `t1`, null disables **Reset**. Null **Readout** or **Pi Pulse** fails during selection, before overrides. Null sweep Stop/Expts, Reps or Rounds leaves a blank; required blanks left after overrides fail readiness before Run. Null **Dual exponential** or **Skip leading points** fails at Analyze.
