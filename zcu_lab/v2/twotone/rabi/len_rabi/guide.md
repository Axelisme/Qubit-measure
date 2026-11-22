**Readers:** operator, agent. Sweep pulse duration at fixed gain and measured carrier to calibrate π and π/2 lengths.

## Sequence and key settings

Optional **Reset** and preparation → length-swept **Rabi Pulse** → calibrated **Readout**. Check the carrier, gain and readout timing before Run. A prior π pulse is not required if you set these directly.

- **Pulse length** starts at `0.03 µs…4 × stored X180 waveform length`. No numeric length gives a 1 µs stop; a stored invalid length leaves Stop empty. Enter a positive stop long enough to see at least one period.
- Use an editable waveform length. Run rejects an arbitrary asset before scanning or plotting; the sweep does not resize or resample its saved samples. Const and flat-top sweep on the FPGA; other editable envelopes are stepped by the host.
- **Relax delay** starts at `5 × qubit.t1` (150 µs if absent). Use **Fit decay envelope** if the oscillation damps. Enable **Fit phase offset** if the data do not match a fixed 0° or 180° phase.

See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Both plots show projected mean IQ (a.u.), not population, versus pulse length (µs). One bar is one shot SEM: retained-shot sample SD on the plotted axis divided by √(retained shot count). It is not shot spread, round-to-round scatter or fit error. Bars do not weight the fit; mean-only files load and fit without bars.

Analyze draws the fitted oscillation and marks π and π/2, the durations for half and quarter of a cycle. Marker shading and ± values are fit standard errors. The title gives Rabi frequency and its fit error. R² compares residuals with measured variation. Check that the fit follows a full oscillation and the markers sit inside the scan; otherwise extend the window or change the fixed gain.

## Next step

Writeback offers `qubit.ext.rabi_freq`; with source settings it also offers `pi_len`/`pi2_len` pulse copies with fitted lengths, separate X180/X90 slot assignments and `qubit.ext.pi_product = source gain × fitted π length`. It does not edit `pi_amp`/`pi2_amp`.

For `time_rabi`, null disables **Reset**. Null **Readout** or **Rabi Pulse** fails during selection, before overrides. Null Frequency, Gain, length Stop/Expts, Reps or Rounds leaves a blank; unresolved required blanks fail readiness before Run. Null **Fit decay envelope** or **Fit phase offset** fails at Analyze.
