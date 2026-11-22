**Readers:** operator, agent. Use AllXY to distinguish drive amplitude and detuning errors by how they distort the expected readout staircase.

## Sequence and setup

Optional **Reset** → one of 21 ordered pairs of I, X90, Y90, X180 and Y180 → **Readout**. Y uses the corresponding X pulse with a 90° phase advance. Calibrate frequency, X90 and X180, and readout first. Verify the pulse modules before Run. **Relax delay** starts at `5 × qubit.t1` (150 µs when T₁ is absent). Each point uses a single gate pair, so average enough **Reps** and **Rounds** to resolve the three levels.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Run and Analyze show projected mean IQ (a.u.) by ordered gate pair, not population. The pairs are categories, so the measured points are not joined. A calibrated sequence gives a staircase for ground, equator and excited states. One error bar is one shot SEM on the plotted IQ axis: retained-shot sample SD divided by √(retained shots). It is not shot spread, round-to-round scatter or fit uncertainty. Mean-only files plot without bars.

Analyze overlays the coherent-error model. **Amplitude error** is relative drive error (+0.02 means 2% too much rotation); `detune_param` is the model's detuning parameter in a.u. The `power_err` and `detune_err` values are mean absolute Bloch-z deviations caused by each error over the 21 pairs (ground +1, excited −1). Ground and excited levels are fitted, not fixed to the measured extremes.

## Next step

Compare residual RMS with those deviations in the same units. A comparable or structured residual means the model does not isolate the errors; it omits decoherence. Check readout, then repeat before interpreting the estimates. Use amplitude Rabi or zig-zag for pulse amplitude and Ramsey for detuning. Use RB to estimate gate errors; AllXY does not report fidelity or write back.
