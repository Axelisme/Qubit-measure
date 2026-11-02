**Readers:** operator, agent. Run mux onetone maps versus the shared coil and write back one participant's flux calibration.

# Mux onetone flux global

## Before Run

Bind alias `mux` to a `group/mux` component. The group must name matching
ordered qubits and resonators. Tone index i always refers to that full group
position, even when participants are reordered or omitted.

The group's `flux_dev` names the connected coil driver.
Choose `flux_unit` to match the driver's stored native unit. No unit conversion,
mode change or output enabling occurs. Confirm limits, wiring, power and output
before running on hardware.

Open or Reset captures identities, aliases and library values. Later edits to
the Library or aliases do not change the captured form. Reset discards form
edits and captures sources again.

## Form

- `participants` selects original tone indices in acquisition order. The tone picker only
  selects which full-list position to edit.
- `flux_dev` starts from the captured group `flux_dev`.
- `flux` is one native-unit sweep shared by all participants, initially 101 points.
  Open/Reset starts with all Qs selected. It projects each Q's `{offset, period}`
  onto the group's `flux_dev`, using other devices' detached values.
  Missing other device values contribute zero; missing offset or selected period
  leaves the required endpoints blank.
  It takes the union of expanded actual half-to-integer intervals.
  Missing selected calibration leaves both endpoints blank; invalid contributors raise an error.
  Changing participants later does not recompute these starting values.
- `settle` is the wait in seconds after all DC writes at each flux point.
- `reset` and `readout` contain the optional common reset and full mux readout.
- `tones` is one fixed identity list in original readout order, with no drive.
- `df` supplies a common nominal MHz offset for every selected readout tone.
  Each offset rebuilds reset and readout in this experiment because the mux
  firmware cannot sweep tone frequencies in hardware.

Readout uses the mux onetone ringup-copy policy. A captured slot retains its
mixer and ADC window; five ringup times are added to trigger coverage. Without
a slot, generator and ADCs come from group wiring, gain starts at 0.1, mixer
comes from group `mixer_freq`, and the ADC window starts at 1.0 µs. Trigger is
group `time_of_fly + 5/(2π min κ)`; pulse length covers trigger plus window.
Missing group flight time or any κ leaves trigger and length required and blank.

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
   the coil source. Missing DC calibration leaves both shared bounds
   required and blank. Enter deliberate native-unit bounds before Run.
   Fill any missing resonator frequency, mixer, trigger and pulse duration. Ringup timing needs every resonator linewidth and the group flight time; without them enter timing explicitly.
2. **Frequencies without a readout slot.** Open or Reset to capture updated
   frequencies. The readout template takes generator/ADCs from group wiring.
   The ADC window candidate is 1 us. Trigger and pulse duration remain blank when linewidth or group flight time is missing.
   Review gain 0.1 and the DC bounds rather than treating seeds as calibrations.
3. **Captured readout slot.** Open or Reset after assigning the group readout.
   The form copies it, rebuilds full group tones and adds ringup to its trigger while preserving its mixer and ADC window.
   Confirm frequency and DC scan bounds. Refresh preserves your edits and the
   original seed provenance while recapturing validation sources. Reset
   discards edits and captures sources again.
4. **Agent recipe.** Use `mux_onetone_flux_global` with explicit
   missing DC bounds. Its inputs must describe the same full identities,
   native units and frequency ranges as this form. After acquisition, inspect
   each PendingAnalysis with `tab_interact`, move the native lines and send
   `done`. Only then use `answer` for the separate writeback question.
   The [recipe guide](../../../recipes/mux-guide.md#mux-flux-maps) lists inputs,
   per-Q `flux_half`/`flux_integer` preselections and selected-item writeback.

## Typed cfg and form paths

Python cfg and saved cfg share a flat layout. There is no `spectrum` wrapper.
Set `reset`, full `readout`, full `tones`, `df` and `participants: list[int]`
at top level.
Open/Reset captures the Aliases row `participants`, for example `[Q3, Q1]`,
and maps it against the captured `$mux.qubits` order to original indices `[2, 0]`.
Alias list elements may use `$` links. A missing, unbound or empty row selects all
original indices in order. An unbound element in a nonempty list rejects Open/Reset.
An out-of-group name stays visible as an invalid Participants draft until corrected.
Changing aliases requires Reset or reopening; ordinary participant edits keep tone values.

Editing participants preserves the fixed tone identities and the shared
frequency and DC sweep values. Readout identity remains required for every tone.
All readout identities stay required, including inactive tones.
`reset` is optional; `readout` is the complete numeric mux
readout. In the form, module selectors expose inline fields below
`readout.module`; this selector wrapper is not part of Python cfg.
`df` is a SweepCfg of common nominal MHz offsets.
`tones[i]` contains only its fixed name. `flux_dev` names the coil; `flux` is a SweepCfg in native DC units.
`flux_unit` is `A`, `V` or `native`; no conversion occurs.
`reps` and `rounds` are positive counts. `initial_delay` and `relax_delay`
are finite nonnegative microseconds. `settle` is finite nonnegative seconds.
Every SweepCfg supplies `start`, `stop`, positive `expts` and
`step = (stop-start)/(expts-1)`; a single point requires equal bounds and zero
step. Form/recipe range editors use `n` for that point count and derive step.

The form orders reset, readout, tone identities, flux device and sweep,
frequency offset sweep, participants, flux unit, settling, then reps, rounds,
initial delay and relaxation delay. Source descriptions distinguish captured settings,
derived candidates and missing required values. Generated candidates are
marked uncalibrated; a stored value alone is not evidence of calibration.
All full-list entries must validate even when not selected for acquisition.
Generated seeds never imply calibration approval. Inspect source descriptions
and required-field errors before Run.

## Run and Stop

Run sets the coil at each flux point and waits for settle. It then scans the
frequency offsets, building reset followed by mux readout at each offset. Each `flux_i` figure shows absolute MHz
against that tone's DC coordinate. Sources retain their final written settings.
Run does not restore bias automatically.

A flux point is published only after its complete inner spectrum returns.
Cancelling an inner acquire leaves the entire current flux point NaN, including
any data already collected there. The scan continues with the next flux point.
To stop the whole scan, cancel the Run scope. The GUI Stop button does this;
it interrupts ramps, settle and acquisition. Earlier completed points remain,
and the interrupted point and later points stay NaN. Analyze needs at least
five completed flux points and at least two distinct frequency values.
Device and acquisition failures are errors, not empty
successful maps.

## Analyze and Writeback

The Analysis parameter `tone` is an original group tone index, not the
participant-list position. It initially selects the first acquired participant.
Choose another acquired index and click Analyze to calibrate that Q.
The shared picker displays GHz. Acquisition and loaded frequency coordinates
remain MHz; the adapter converts only the picker input.
The picker starts in magnitude mode.
Move the half-flux and integer-flux lines, then click Done. The existing picker
also provides Swap Lines, Auto Align and Cancel. Period is twice the absolute
half-to-integer separation. The summary includes the original index and Q name.

Review the resulting proposals before applying Writeback. Destinations use the
Run/Load capture, not the current form or aliases.
The proposals separately update that Q's `flux.period.<scanned device>` and
`flux.offset`. The device is the actual `cfg.flux_dev`, not a reserved `coil` alias.
Period updates its magnitude and preserves its sign, initially positive.
The default-selected offset uses `0.5 - half_fit / period_s - Σ V_o / period_o`.
Other values come from the Run-start snapshot; the swept value is replaced by
half_fit once. The note lists other present calibrated devices' values and
contributions, including zero values. Missing devices contribute zero and retain
their periods. Other periods and Qs are preserved. Simultaneous scans do not
block the proposal; the experiment author must judge that calibration.
The accompanying ordinary record stores the raw picked half/integer/positive period, scanned device, complete connected-device Run snapshot and original Run parameter frame.
Review the period, offset and record items separately. Accepting period does not
require accepting offset. A missing whole Run snapshot blocks calibration;
missing individual device values contribute zero. Acceptance never reads live
devices or current parameters.
Repeat Analyze and Writeback for each acquired participant. Cancel produces no
new calibration. Missing historical cfg prevents identity-dependent analysis;
missing parameter capture permits picking but yields no writeback proposals.

## Saved data

Native data keeps original tone indices, native DC coordinates per tone,
absolute MHz frequency coordinates per tone, and complex IQ shaped
`(participants, flux points, frequency points)`. Frequencies use Hz on disk and
MHz after loading. Stop gaps remain NaN after native round-trip.

Cfg schema 5.0 stores the independent flat cfg with top-level `df`. Loading
schema 4.0 keeps the data and plots available but does not restore its cfg.
The load reports a cfg schema mismatch. No automatic cfg conversion occurs.

Labber export splits each tone's three-dimensional independent payload into
one two-dimensional log per first-axis slice in the same file. The grouped
reader returns those two-dimensional slices, not a reconstructed tone cube.
Inspect each output status before treating a Save as complete. Load the native
`data.h5` for analysis and writeback.
