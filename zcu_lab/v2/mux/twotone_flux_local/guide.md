**Readers:** operator, agent. Run mux twotone maps versus each qubit's own DC line and write back one participant's flux calibration.

# Mux twotone flux local

## Before Run

Bind alias `mux` to a `group/mux` component. The group must name matching
ordered qubits and resonators. Tone index i always refers to that full group
position, even when participants are reordered or omitted.

Each qubit's `flux_source` names its own connected DC driver. All local sources must be distinct.
Choose `flux_unit` to match the driver's stored native unit. No unit conversion,
mode change or output enabling occurs. Confirm limits, wiring, power and output
before running on hardware.

Open or Reset captures identities, aliases and library values. Later edits to
the Library or aliases do not change the captured form. Reset discards form
edits and captures sources again.

## Form

- `participants` selects Q names in acquisition order. The tone picker only
  selects which full-list position to edit.
- `tones[i].flux_dev` starts from that Q's captured `flux_source`.
- Each `tones[i].sweep` has independent native-unit Start and Stop.
  `flux_expts` is the one point count shared by all local lines, initially 101.
  Lines advance together by point index, not as a Cartesian product.
- Start is `1.1*Q.flux_int - 0.1*Q.flux_half`; Stop is
  `1.1*Q.flux_half - 0.1*Q.flux_int`. Missing calibration leaves required
  endpoints blank. Unselected lines are validated but never written.
- `settle` is the wait in seconds after all DC writes at each flux point.
- `spectrum` contains the inner mux settings, averaging and readout.
  Each `spectrum.tones[i].sweep` has drive-frequency Start/Stop in MHz.
  `spectrum.expts` is the shared hardware point count. Inner participants
  displays the fixed full identity list; outer participants supplies the Run
  selection. All selected drives sweep together in one hardware loop.

Readout references the captured group readout slot when present. Without a
slot, generator and ADCs come from group wiring, mixer from group
`mixer_freq`, and gain starts at 0.1. Frequency uses each resonator's calibration;
pulse and ADC durations remain required and blank. Drive defaults use each
Q's probe template and calibrated frequency/linewidth.

These initial values are editable candidates, not confirmed calibrations.
Averaging starts at 1000 reps and 10 rounds. Reduce both and the sweep point
counts for an offline smoke.

## Run and Stop

Run sets only selected local sources at each flux point, waits for settle, then runs
the existing mux twotone spectrum. Each `flux_i` figure shows absolute MHz
against that tone's DC coordinate. Sources retain their final written settings.
Run does not restore bias automatically.

Stop interrupts ramps, settle and acquisition. A flux point is published only
after the complete inner spectrum returns. Earlier completed points remain;
the interrupted point and later points stay NaN. Analyze needs at least five
completed flux points. Device and acquisition failures are errors, not empty
successful maps.

## Analyze and Writeback

The Analysis parameter `tone` is an original group tone index, not the
participant-list position. It initially selects the first acquired participant.
Choose another acquired index and click Analyze to calibrate that Q.
The picker starts in phase-aware mode, as in twotone/fluxdep.
Move the half-flux and integer-flux lines, then click Done. The existing picker
also provides Swap Lines, Auto Align and Cancel. Period is twice the absolute
half-to-integer separation. The summary includes the original index and Q name.

Review the resulting proposals before applying Writeback. Destinations use the
Run/Load capture, not the current form or aliases.
Only that Q's `flux_half`, `flux_int` and `flux_period` are proposed.
Repeat Analyze and Writeback for each acquired participant. Cancel produces no
new calibration. Missing historical cfg prevents identity-dependent analysis;
missing parameter capture permits picking but yields no writeback proposals.

## Saved data

Native data keeps original tone indices, native DC coordinates per tone,
absolute MHz frequency coordinates per tone, and complex IQ shaped
`(participants, flux points, frequency points)`. Frequencies use Hz on disk and
MHz after loading. Stop gaps remain NaN after native round-trip.

Labber export currently does not support this three-dimensional independent
grid. Save reports that output as failed while native data and other successful
outputs remain saved. Inspect each output status before treating a Save as
complete. Load the native `data.h5` for analysis and writeback.
