**Readers:** operator, agent. Track a resonator notch versus DC bias to locate fluxonium half- and integer-flux symmetry points.

# One-tone flux dependence

Set DC bias → sweep absolute readout RF → at each frequency play one readout pulse and integrate its demodulated IQ. Average the shots at each frequency before moving to the next bias.

## Set the sweep

- Bind qubit and resonator. Use [lookback](../../lookback/guide.md) for flight time and [one-tone spectroscopy](../freq/guide.md) for a low-gain center and linewidth. Check the Readout channels and ADC window. **Trig offset** starts at flight time + `5/(2π·κ)` µs for κ in MHz, or flight time alone without κ. Pulse length covers trigger plus **RO length** (template length minus 0.03 µs), and is at least the template length. Without flight time, trigger and pulse length are required blanks.
- **Post-delay** lets the cavity empty. Start at `5/(2π·κ)` µs; to calibrate this wait, compare herald → nominal Xπ with a longer wait.
- **Flux-bias device** starts from the qubit binding and **Bias unit** from that source's captured native unit. Verify both. **Flux bias** starts from integer toward half with 10% padding on both sides. The saved offset and periods are evaluated at the other devices' captured values, with missing values counted as zero. Missing offset or selected period leaves the bounds blank. Use approved device-native bounds, not Φ₀.
- **Readout RF frequency** starts at the captured resonator center ± one linewidth. Use absolute MHz and keep the notch inside the window over the whole bias span. Keep Readout gain below the drive-dependent frequency shift.

For tab setup, refreshing sources, applying and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the map and pick symmetry points

Live shows `|mean(I + iQ)|` (a.u.) versus bias and absolute RF (MHz), with a cut at the latest bias. Blank columns are unmeasured, not zero. Neither panel shows population or shot SEM.

In the picker, track the notch. Drag the red line to half flux and blue to integer flux. Frequency is shown in GHz there and in MHz on the final map. Lower mirror loss means better local reflection symmetry; it does not identify half versus integer. The initial lines use the analyzed Run or Load capture and other devices' Run-start values, or picker defaults if the selected calibration is absent.

**Done** plots the accepted lines. **Bias period** is twice their separation in source units, not a fitted parameter with a standard error. If a pick falls at the scan edge, extend the bias range. If the notch is clipped, extend the RF window before picking. Without restorable cfg the final plot uses device units and omits Reps × Rounds.

## Use the calibration

With a compatible capture and Run device snapshot, writeback offers the swept device's signed period, qubit offset, and a raw calibration record. The period keeps its old sign, initially positive. Offset is preselected and uses `0.5 − half_fit/period_s − Σ V_o/period_o`, where the sum uses only other calibrated devices at Run start. Missing values count as zero. The preview lists their contributions; review period and offset separately.

Recompute the integer bias from the accepted calibration at the other devices' current values. Verify it with a narrow one-tone sweep before two-tone spectroscopy.

## Recipe inputs

`onetone_spectrum_over_flux`: the schema requires non-null `flux_device`, `flux_unit`, `flux_range` and `flux_points`. Before Run, the recipe rejects identical bias endpoints or a unit that does not match the usable cached device unit, including explicit `native` where applicable.

Supply `center_mhz` and `span_mhz` together or omit both. Only one supplied returns `needs_parameters` before Run; if either is null with both supplied, both endpoints clear. Null `gain`, `freq_points`, `reps` or `rounds` passes to the Run Form; any unresolved required input fails readiness after overrides. Null `readout_ref` is rejected selecting the required Readout, before Run.
