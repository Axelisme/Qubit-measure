## Behavior

Measure relaxation during a fast-flux pulse at each programmed gain. Each sequence
resets, applies a calibrated pi pulse, applies the flux pulse, then reads out.
The gain and pulse-length sweeps produce complex IQ shaped (gain, length).
Gain is normalized DAC amplitude, not calibrated flux. Length and fitted T1 use us.

## Open and Reset

Open and Reset capture the current aliases and library. The pi pulse uses
`$qubit.module.x180` when available, otherwise the `pi_pulse` template.
Readout uses `$qubit.module.readout` or the `readout` template. Missing reset
disables that optional module. Missing required calibration leaves fields Invalid.

The `flux_pulse` role starts with a const waveform, gain 1.0, frequency 0 MHz
and Nyquist zone 2. Its channel comes from `$qubit.wiring.lo_flux_ch`.
An unresolved channel stays Invalid; there is no fallback channel or flux slot.
This form seeds a 0.1 us waveform duration. These initial values are not a flux
calibration. Check the channel, waveform, timing and wiring before Run.

The gain sweep initially contains one point at zero. **Set the gain sweep**
to the range you intend to measure. The core replaces the flux pulse's gain
with this sweep, so editing only the module gain does not change scanned gains.
The core also replaces its pulse length with the length sweep. Use a waveform
with a sweepable duration, such as const or flat_top, not a fixed arb asset.

Length starts at 0 and ends at 5 times captured qubit T1 with 101 points.
Relax delay uses the same endpoint. Missing or null T1 uses a 30 us time scale,
giving 150 us. Stored invalid T1 leaves the derived fields empty.
These editable timing estimates are not written back as calibration.
Reps start at 1000 and rounds at 100. Reset refreshes sources and discards edits;
editing aliases alone does not change an already open form.

## Run and analyze

Calibrate readout and the pi pulse first. Choose a safe flux gain range and a
length window that resolves the decay. Inspect the live gain-versus-time map.
A zero-gain run is a wiring and acquisition check, not evidence of flux response.

Analyze fits each gain's decay, estimates a common baseline from accepted
preliminary fits, then refits with that baseline fixed. The named `fit` figure
overlays T1 on the IQ map. Rejected T1 estimates appear as null in the summary;
their initialized zero errors are not valid uncertainty estimates. A flat,
short or noisy trace may not support a fit. Inspect the figure rather than
treating a returned value as automatic acceptance.

There is no writeback proposal. This experiment measures a gain-dependent curve,
not a single operating-point T1. Native Save preserves the core's gain and time
axes, complex IQ and source cfg. Load and Analyze use that saved source, not
the current form, alias table or library. Time is stored in seconds and restored
to us. Analyze can use valid saved data even when its cfg cannot be restored.
