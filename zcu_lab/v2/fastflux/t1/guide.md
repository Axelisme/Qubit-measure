# Fast-flux T1

**Readers:** operator, agent. Measure relaxation during fast-flux pulses from the GUI or MCP and inspect gain-dependent T1 estimates.

**Last updated:** 2026-10-21

## What this measures

Measure relaxation during a fast-flux pulse at each programmed gain. Each sequence
resets, applies a calibrated pi pulse, applies the flux pulse, then reads out.
The gain and pulse-length sweeps produce complex IQ shaped (gain, length).
Gain is normalized DAC amplitude. Converting it to flux requires a separate
calibration. Length and fitted T1 use us.

## Open and Reset

Open and Reset capture the current aliases and library. The pi pulse uses
`$qubit.module.x180` when available, otherwise the `pi_pulse` template.
Readout uses `$resonator.module.readout` or the `readout` template. Missing reset
disables that optional module. Missing required calibration leaves fields Invalid.

The Flux pulse module slot uses the `flux_pulse` template. It starts with a const
waveform, gain 1.0, frequency 0 MHz and Nyquist zone 2.
Its channel comes from `$qubit.wiring.lo_flux_ch`.
An unresolved channel leaves the form Invalid until you supply a confirmed channel.
This form seeds a 0.1 us waveform duration. These values are editing seeds;
establish the flux calibration separately. Check the channel, waveform, timing
and wiring before Run.

The gain sweep initially contains one point at zero. **Set the gain sweep**
to the range you intend to measure. Run replaces the flux pulse's gain
with this sweep. Set scanned gains in the gain sweep field.
Run also replaces its pulse length with the length sweep. Use a waveform
with a sweepable duration, such as const or flat_top. A fixed arb asset has a
fixed duration and is unsuitable for this length sweep.

Length starts at 0.1 us and ends at 5 times captured qubit T1 with 101 points.
The positive start avoids a zero-length pulse, which the generator compiler rejects.
Check the minimum duration and sweep spacing on the actual generator's clock grid.
Relax delay uses the same endpoint. Missing or null T1 uses a 30 us time scale,
giving 150 us. Stored invalid T1 leaves the derived fields empty.
These editable timing estimates affect this form only; stored calibration retains
its values. Reps start at 1000 and rounds at 10. Reset refreshes sources and
discards edits. After editing aliases, use Reset to refresh an already open form.

## Run and analyze

Calibrate readout and the pi pulse first. Choose a safe flux gain range and a
length window that resolves the decay. Inspect the live gain-versus-time map.
A zero-gain run checks wiring and acquisition only. Measuring flux response
requires an appropriate gain range.

Analyze fits each gain's decay, estimates a common baseline from accepted
preliminary fits, then refits with that baseline fixed. The named `fit` figure
overlays T1 on the IQ map. Rejected T1 estimates appear as null in the summary;
their initialized zero errors are placeholders only. A flat, short or noisy trace
may yield a rejected fit. Inspect the figure before accepting a returned value.

This experiment measures a gain-dependent curve. Analysis produces a figure and
summary; the stored operating-point T1 retains its value. Save preserves gain
and time axes, complex IQ and source cfg. Load and Analyze use that saved source;
the current form, alias table and library remain separate. Time is stored in
seconds and restored to us. Analyze can use valid saved data even when its cfg
is missing or unusable.
