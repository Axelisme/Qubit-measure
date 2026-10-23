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

- `participants` selects original tone indices in acquisition order. The tone picker only
  selects which full-list position to edit.
- `tones[i].flux_dev` starts from that Q's captured `flux_source`.
- Each `tones[i].sweep` has independent native-unit Start and Stop.
  `flux_expts` is the one point count shared by all local lines, initially 101.
  Lines advance together by point index, not as a Cartesian product.
- Start is `1.1*Q.flux_int - 0.1*Q.flux_half`; Stop is
  `1.1*Q.flux_half - 0.1*Q.flux_int`. Missing calibration leaves required
  endpoints blank. Unselected lines are validated but never written.
- `settle` is the wait in seconds after all DC writes at each flux point.
- `modules` contains the optional common reset and full mux readout.
- `frequency` contains only frequency-scan inputs.
  Each `frequency.tones[i].sweep` has drive-frequency Start/Stop in MHz.
  `frequency.expts` is the shared hardware point count. The frequency tone picker retains all original names;
  top-level participants supplies the Run selection. All selected drives sweep together in one hardware loop.

Readout references the captured group readout slot when present. Without a
slot, generator and ADCs come from group wiring, mixer from group
`mixer_freq`, and gain starts at 0.1. Frequency uses each resonator's calibration;
pulse and ADC durations remain required and blank. Drive defaults use each
Q's probe template and calibrated frequency/linewidth.

These initial values are editable candidates, not confirmed calibrations.
Averaging starts at 1000 reps and 10 rounds. Reduce both and the sweep point
counts for an offline smoke.

## Recipe input semantics

Omitted Form inputs retain captured values. Explicit null clears a field or
range bounds. The Form chooses sources at Open/Reset; recipes do not offer
separate library/inline input lists. Overrides use dotted paths, run in order
and run last. Decimal indices keep their original complete-list positions.
The recipe calls `check_ready()` before acquisition and reports all owner paths
and reasons without rolling back edits. Analysis null goes to the picker owner.

## Starting scenarios

1. **Blank kickstart.** Confirm group wiring, resonator nominal frequencies and
   the six distinct local sources. Missing DC calibration leaves each local bounds
   required and blank. Enter deliberate native-unit bounds before Run.
   Fill each drive-frequency range if qubit frequency or linewidth is absent. Confirm readout mixer, ADC channels, trigger and pulse/ADC durations; a blank readout template does not invent timing.
2. **Frequencies without a readout slot.** Open or Reset to capture updated
   frequencies. The readout template takes generator/ADCs from group wiring.
   Readout timing remains required. Drive bounds derive from each qubit frequency and linewidth only when both exist.
   Review gain 0.1 and the DC bounds rather than treating seeds as calibrations.
3. **Captured readout slot.** Open or Reset after assigning the group readout.
   The form references the stored readout without replacing its timing, gains or tones.
   Confirm frequency and DC scan bounds. Refresh preserves your edits and the
   original seed provenance while recapturing validation sources. Reset
   discards edits and captures sources again.
4. **Agent recipe.** Use `mux_twotone_flux_local` with explicit
   missing DC bounds and drive-frequency bounds. Its inputs must describe the same full identities,
   native units and frequency ranges as this form. After acquisition, inspect
   each PendingAnalysis with `tab_interact`, move the native lines and send
   `done`. Only then use `answer` for the separate writeback question.
   The [recipe guide](../../../recipes/mux-guide.md#mux-flux-maps) lists inputs,
   per-Q `flux_half`/`flux_integer` preselections and selected-item writeback.

## Typed cfg and form paths

Python cfg and saved cfg share a flat layout. There is no `spectrum` wrapper.
Set `participants: list[int]`, full `tones`, and `modules` at top level.
Open/Reset fills selected tones' drive and per-tone sweep defaults and sets
these fields to None on other tones. The nullable fields remain editable.
Open/Reset captures the Aliases row `participants`, for example `[Q3, Q1]`,
and maps it against the captured `$mux.qubits` order to original indices `[2, 0]`.
Alias list elements may use `$` links. A missing, unbound or empty row selects all
original indices in order. An unbound element in a nonempty list rejects Open/Reset.
An out-of-group name stays visible as an invalid Participants draft until corrected.
Changing aliases requires Reset or reopening; ordinary participant edits keep tone values.

Editing participants preserves every tone value. Fill any newly selected tone's
missing settings at the indicated tone path. Removing a tone keeps its values.
All readout identities and local flux_dev names stay required. Local driver
uniqueness is checked only for selected tones.
`modules.reset` is optional; `modules.readout` is the complete numeric mux
readout. In the form, module selectors expose inline fields below
`modules.readout.module`; this selector wrapper is not part of Python cfg.
`frequency.tones[i]` contains the fixed name, numeric drive pulse and `sweep.start/stop` in absolute MHz. `frequency.expts` is the shared positive hardware count.
`tones[i]` contains its fixed name, `flux_dev`, and `sweep.start/stop` in native DC units. `flux_expts` is their shared positive count.
`flux_unit` is `A`, `V` or `native`; no conversion occurs.
`reps` and `rounds` are positive counts. `initial_delay` and `relax_delay`
are finite nonnegative microseconds. `settle` is finite nonnegative seconds.
Every SweepCfg supplies `start`, `stop`, positive `expts` and
`step = (stop-start)/(expts-1)`; a single point requires equal bounds and zero
step. Form/recipe range editors use `n` for that point count and derive step.

The form orders participants, tone identities, modules, frequency/DC scans,
averaging, then timing. Source descriptions distinguish captured settings,
derived candidates and missing required values. Generated candidates are
marked uncalibrated; a stored value alone is not evidence of calibration.
All full-list entries must validate even when not selected for acquisition.
Generated seeds never imply calibration approval. Inspect source descriptions
and required-field errors before Run.

## Run and Stop

Run sets only selected local sources at each flux point, waits for settle, then runs
the existing mux twotone spectrum. Each `flux_i` figure shows absolute MHz
against that tone's DC coordinate. Sources retain their final written settings.
Run does not restore bias automatically.

Stop interrupts ramps, settle and acquisition. A flux point is published only
after the complete inner spectrum returns. Earlier completed points remain;
the interrupted point and later points stay NaN. Analyze needs at least five
completed flux points and at least two distinct frequency values.
Device and acquisition failures are errors, not empty
successful maps.

## Analyze and Writeback

The Analysis parameter `tone` is an original group tone index, not the
participant-list position. It initially selects the first acquired participant.
Choose another acquired index and click Analyze to calibrate that Q.
The shared picker displays GHz. Acquisition and loaded frequency coordinates
remain MHz; the adapter converts only the picker input.
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
