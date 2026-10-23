# Mux recipes

**Readers:** operator, agent. Run mux resonator, qubit and DC flux recipes, save results and review calibration proposals.

## mux_onetone

`mux_onetone` opens `mux/onetone` for a common-offset resonator scan.
It does not require qubit drives. The seven qubit recipes below have different
participant and sweep contracts.

| Input | Meaning | If omitted |
| --- | --- | --- |
| `target`, `reuse_tab_id` | Mux alias override and idle matching tab to Reset | Use global aliases and open a tab |
| `participants` | Unique original readout indices in execution order | Keep Active tones |
| `df` | `{start, stop}` common offset in MHz, added to each base frequency | Keep the Form range |
| `n` | Shared offset point count, at least two | Keep Expts |
| `reps`, `rounds` | Positive acquisition counts | Keep Form values |
| `model_type` | Resonator model: `hm`, `t` or `auto` | Keep analysis settings |
| `fit_bg_amp_slope`, `fit_bg_phase_curvature` | Fit the background slope or phase curvature | Keep analysis settings |
| `edelay` | Manual cable-delay seed in inverse MHz | Keep analysis settings |
| `overrides` | Ordered final `{path, value}` edits using dotted paths | No extra edits |

Indices address the complete captured readout list, not display labels or
current global aliases. Empty, duplicate, negative and out-of-range indices
return field errors before acquisition.
The Form chooses captured sources at Open or Reset. The recipe sets participants,
df/n and counts, then applies overrides in order and calls `check_ready()`.
Omission retains a Form input. Explicit null clears it, including range bounds.
Analysis inputs also retain settings when omitted; null goes to the analysis owner.
For `target` and `reuse_tab_id`, null keeps the global alias and opens a new tab.

Analyze reports each original index, bound name, display label, frequency,
kappa, fit_quality, and any per-tone failure. `resonators_diagnostic` explains
missing identity bindings; it does not discard module proposals. Labels may
repeat and do not establish identity. Fits retain native analysis order.
The recipe asks before applying only selected proposals and saves once after
acquisition. Skipping writes nothing. `writeback_applied` records confirmed
applied items, not merely an accepted answer.

Read the [adapter guide](../v2/mux/onetone/guide.md) for source defaults,
fit diagnostics, timing checks and proposal destinations.

These seven recipes measure selected qubits with simultaneous mux readout, save
raw data and fits, and ask before applying calibration updates.

## Mux flux maps

Four recipes open the matching flat-cfg adapters. All four take original full-tone indices as
participants. `mux_onetone` uses the same index contract for its readout list.

| Recipe | Frequency input | DC input |
| --- | --- | --- |
| `mux_onetone_flux_global` | `df:{start,stop}` common MHz offsets, `n` | `flux:{start,stop}` coil native values, `flux_n` |
| `mux_onetone_flux_local` | `df:{start,stop}` common MHz offsets, `n` | `flux_sweeps:[{start,stop}]` native local values, `flux_n` |
| `mux_twotone_flux_global` | `sweeps:[{start,stop}]` absolute MHz per drive, `n` | `flux:{start,stop}` coil native values, `flux_n` |
| `mux_twotone_flux_local` | `sweeps:[{start,stop}]` absolute MHz per drive, `n` | `flux_sweeps:[{start,stop}]` native local values, `flux_n` |

Both list inputs cover every original tone, including nonparticipants, in full
group order. Neither accepts a per-tone point count. `n` sets
`frequency.df.n` in onetone or `frequency.expts` in twotone.
`flux_n` sets `flux.n` globally or `flux_expts` locally.
Range editors derive the core SweepCfg step and expts. Counts must be positive;
a one-point scan requires equal bounds. Picking needs at least five completed
DC points and two distinct frequency values.

`target` overrides this Reset's mux alias. `reuse_tab_id` resets an idle
matching tab. Omitting either control, or passing null, uses global aliases and
opens a new tab. `participants` is an optional ordered list of unique original
tone indices. Omission retains the Form selection; null clears it.
The Form chooses sources. Range/count edits follow, then `overrides` applies last.
Dotted paths always address original full-list positions. Omitted inputs keep
Form values; null clears them. All readout identities must validate before Run.
Drive/sweep values validate only for selected tones, and inactive drafts stay
in the form. A range list with the wrong length returns `needs_parameters` with
the missing or extra original positions before acquisition.

`flux_unit` optionally labels A, V or native. Omission keeps the label; null clears it.
The device owner checks it against each selected source. No values are rescaled.
Global source identity starts at the captured `$mux.flux_source`; local sources
start at each captured `Q.flux_source`. Use overrides for intentional device,
averaging or timing edits. There is no nested `spectrum` cfg.

For example, a three-Q group can scan Q3 then Q1 with this onetone global input.
The numeric bounds are illustrative, not device limits or known calibration.

```json
{
  "target": "MG",
  "participants": [2, 0],
  "df": {"start": -2.0, "stop": 2.0},
  "n": 31,
  "flux": {"start": -0.01, "stop": 0.01},
  "flux_n": 21,
  "flux_unit": "A",
  "flux_half": {"Q3": 0.004},
  "flux_integer": {"Q3": -0.002},
  "overrides": [
    {"path": "reps", "value": 10},
    {"path": "rounds", "value": 1}
  ]
}
```

Each picker uses the selected Q's original tone index, so Q3/Q1 is 2/0, not 0/1.
`flux_half` and `flux_integer` are optional Q-name-to-native-value maps.
Omit any Q or either line to use its Run-captured calibration, then the picker's
default if absent. Keys must name final selected participants. Explicit values
only seed this analysis, never Run cfg or calibration. The owner field for
`flux_integer` is `flux_int`. The picker may fold initial lines near the scan
center using its period rule. Review its actual committed state.

### Starting with missing calibration

1. With blank kickstart wiring, use onetone first. Supply deliberate global
   `flux` or full-list local `flux_sweeps` when DC calibration is missing.
   Complete required resonator frequencies, mixer and readout timing using
   Form selectors and final overrides. Wiring and nominal values alone do not make
   the Form valid. Do not infer bounds from simulator internals.
2. With frequencies but no readout slot, Open or Reset creates the captured
   template. Onetone seeds ADC window 1 us and gain 0.1. Missing linewidths or
   group flight time leave trigger and pulse length required. Twotone requires
   readout pulse length and ADC duration; it also requires every drive and MHz
   range. Supply full-list `sweeps` if frequency/linewidth calibration is absent.
3. With a readout slot, confirm its captured contents and remaining errors.
   Onetone copies group tones and applies ringup timing; twotone keeps the slot.
   Slot existence does not prove complete drives, frequency bounds or DC bounds.
   Local bounds derive from each Q's captured flux_int/flux_half. Global bounds
   use the captured union of all initial participants' coil intervals. Missing
   calibration keeps endpoints required. Selection changes do not recompute seeds.
4. For an agent recipe, pass explicit missing inputs in the shapes above.
   `needs_parameters` includes owner paths and reasons. Fix those fields rather
   than retrying Run unchanged. Inspect `get_cfg` input, values, errors and
   diagnostics to distinguish missing calibration from an invalid source.

Open and Reset capture fresh sources. Refresh preserves authored inputs,
explicit selections and resolved seeds; it does not refill defaults. After
changing aliases, library or calibrations, Reset to use them. Reset discards
unsaved Form edits. All four forms show participants, tone identities, modules,
frequency/DC scans, averaging, then timing. Frequency labels use MHz, timing
uses us except settle in s, and DC coordinates use the chosen flux_unit.
Seed descriptions expose source paths and mark generated candidates
uncalibrated. A captured slot is not marked measured merely because it exists.

### Picker handoff and selected writeback

After the single acquisition, the recipe starts one tracked interactive
Primary analysis per participant. Updates and Analyze use that same operation.
Each Q receives both `flux_half` and `flux_int`. An unspecified line is sent as
null so the owner uses that Q's Run capture, not the previous Q's explicit line.
This per-Q reset is required even when the caller omits both maps. Other
analysis updates omit only KEEP and pass explicit null unchanged.
When the reply contains `PendingAnalysis`, use `tab_interact` without payload
to inspect state and available commands. Move the native half/integer lines
with its commands and finish with `done`. This settles that analysis and resumes
the recipe. A following picker handoff means the next participant is ready.
A writeback question instead includes a request id and proposals for the Q that
just completed. `answer` answers only that question; it cannot move picker lines
or finish analysis.

Review each proposal's original index, Q and destination. Respond with selected
items or decline. The recipe applies only those selected IDs and reports
`writeback_applied` from the owner receipt, not from accepted alone.
Global analysis proposes one complete `MG.coil_flux.<Q>` record containing
`half`, `period`, `int`. It can create that Q in an empty mapping and preserves
other Q records. Local analysis proposes that Q's `flux_half`, `flux_period`,
`flux_int` scalars. Destinations use Run capture, never current aliases.

The report retains completed Q rows in participant order, original index, Q,
native flux_half/flux_integer/flux_period and per-Q writeback_applied.
Expected analysis failure stops later analyses, returns analysis_failure and
keeps earlier rows. Other failures propagate. Save runs once in the finalizer
after acquisition returns, including analysis or writeback errors. Repeated
Primary analysis replaces the previous pane, so Save contains raw maps and the
owner's current completed artifacts, not a promise of one saved pane per Q.
The report retains earlier Q outcomes.

Native data preserves tone-major three-dimensional IQ and DC/MHz coordinates.
Labber cannot export this independent grid. Inspect per-output statuses in
`runs`; native success and Labber failure can coexist. Recipe completion is
not proof that every output saved.

Adapter details and GUI paths are in the
[onetone global](../v2/mux/onetone_flux_global/guide.md),
[onetone local](../v2/mux/onetone_flux_local/guide.md),
[twotone global](../v2/mux/twotone_flux_global/guide.md) and
[twotone local](../v2/mux/twotone_flux_local/guide.md) guides.

## Before running

Open an entry and point. Bind `$mux` to a MuxGroup containing the complete ordered
qubit and resonator lists. Each qubit must refer to its matching resonator.
Connect the SoC and any devices needed for the measurement in an authorized
acquisition environment. Recipes do not establish those connections.

A frequency writeback changes `Q*.freq`, not an existing pulse library module.
After accepting a frequency correction, recalibrate the pulse at the new
frequency, for example with Rabi or zigzag. Check the recalibrated gain and
length, save the new numeric pulse module and select it for later experiments.
Changing only the pulse frequency does not calibrate its gain or length.
Pulse modules can come from calibration writeback, New with a template, or
`ml_edit`. Library entries store numbers; creation-time expressions do not stay
bound to future calibration changes. Reset downstream tabs to capture the new
module selection. Historical Run/Load sources retain their original pulses.

Check the group's readout module, drive channels and power before running.
When `$mux.module.readout` is present, the Form uses that library module.
Without it, the Form starts with an inline mux template and group tones.
Complete its required fields before Run. The template is not a hardware-safe
preset.

Selected tones need valid drives. All tones need valid readout identities. Drives use
independent generators, separate from the readout generator. Choose a scan range
for the intended calibration. Recipes retain Form defaults where you omit an
input; they do not infer a new scan range.

Choose the recipe for the measurement:

- [Qubit two-tone spectrum](#mux_twotone_spectrum) scans drive frequency.
- [Amplitude Rabi](#mux_amp_rabi) scans drive gain.
- [Length Rabi](#mux_len_rabi) scans drive duration.
- [T1](#mux_t1) measures energy relaxation.
- [Ramsey](#mux_t2ramsey) measures free-evolution coherence.
- [Echo](#mux_t2echo) measures coherence with a refocusing pulse.
- [Single-shot GE](#mux_singleshot_ge) compares probe-off and probe-on IQ shots.

## Choose inputs

All seven recipes accept the following controls and Form edits:

| Input | What to supply | If omitted |
| --- | --- | --- |
| `target` | MuxGroup component name, such as `MG3`, for this Reset's `$mux` binding | Use global aliases |
| `reuse_tab_id` | An idle matching tab to Reset | Open a new tab |
| `participants` | Nonempty, unique original tone indices in measurement order | Keep Reset defaults |
| `overrides` | Ordered `{path, value}` edits with dotted paths | No extra edits |

Explicit null clears a Form input. Null `participants` does not retain the
selection; it produces a missing selection. Null `target` or `reuse_tab_id`
retains the Open/Reset control behavior shown above. `overrides` cannot be null.

Analysis inputs go to the same Primary operation, not the measurement Form.
Omission retains that tab's analysis setting. Explicit null goes to the analysis
owner, which may reject it for a required setting. These inputs do not change
captured Run settings or writeback policy.

| Recipe | Analysis inputs |
| --- | --- |
| `mux_twotone_spectrum` | `model_type`: `lor` or `sinc`; `plot_fit`: draw the fitted curve |
| `mux_amp_rabi` | `skip`: points omitted before fitting |
| `mux_len_rabi` | `decay`: fit the envelope; `fit_phase`: fit the phase offset |
| `mux_t1` | `dual_exp`: fit two components; `skip`: points omitted before fitting |
| `mux_t2ramsey` | `fit_fringe`: fit oscillations; `fit_phase`: fit their phase offset |
| `mux_t2echo` | `fit_method`: `fringe` or `decay`; `fit_phase`: fit the fringe phase offset |
| `mux_singleshot_ge` | `initial_state`: `ground` or `excited`; `backend`: `pca` or `center`; `logscale`, `align_t1`, `length_ratio` |

The three pulse-scan recipes also accept:

| Input | What to supply | If omitted |
| --- | --- | --- |
| `sweeps` | One `{start, stop}` for every original tone, including nonparticipants | Keep ranges |
| `n` | A positive integer point count shared by all tones | Keep Expts |

Frequency endpoints use MHz, amplitude endpoints normalized gain and length
endpoints us. Each sweep accepts only start and stop; use top-level `n` for the
shared count. A list length mismatch returns `needs_parameters` naming missing
or extra original positions. Null `sweeps` clears every range's bounds; null
`n` clears the common count.

The recipe opens or resets the tab, sets experiment values, applies overrides
in order, then calls `check_ready()`. Duplicate override paths run in order;
the last value wins, including null. Readiness errors list every owner path and
reason. They prevent Run but do not roll back earlier edits. Readout identities
must be valid for all tones; only participant drive/sweep settings are required.
Inactive settings may stay blank.

Paths are dotted strings. Decimal segments address the complete tone list.
For Q1/Q2/Q3, `participants: [2, 0]` measures Q3 then Q1.
Q3 edits still use `tones.2`, not row 0 or the name `Q3`.
For example, set a common point count and override Q3's amplitude range:

```json
{
  "target": "MG3",
  "participants": [2, 0],
  "n": 51,
  "overrides": [
    {"path": "tones.2.sweep.start", "value": 0.0},
    {"path": "tones.2.sweep.stop", "value": 0.3}
  ]
}
```

These bounds illustrate the format, not approved device limits.
Use your entry's names and approved ranges. The Form chooses module sources;
a broken source reports an error rather than falling back to calibration.
There are no generic `library` or `inline` recipe inputs.

To customize source selection, edit the recipe's setup before overrides.
Use `tab.select` to choose an intentional module option.
Before editing a scalar on the current source, read its selector.
If `selector.mode == "ref"`, use
`tab.command("tones.2.drive", "to_inline")` to copy the source first.
Already-inline modules need no conversion. For nested waveform length, read
that waveform's selector and copy only if referenced. This preserves its
family and other values rather than replacing it with a const waveform.
Do not use a scalar override to edit through a reference.

`tab.describe_cfg` lists fields and commands. `tab.read_elements` lists complete
tone identities. Pulse-scan ranges use `start` and `stop`, with common count
`expts`. Delay-scan ranges retain their own common `start`, `stop` and `n`.
Set individual leaves when changing part of a range. Container replacement does
not merge with its previous value. An expression uses `{"expr": "..."}` and
stays an expression in the Form. Reps, rounds, recovery delay and the delay-scan
axis are shared, not per-qubit controls.

## Read conditions and results

Each recipe returns `status`, an authored `report` and `runs`.
The report's `tones` list follows native analysis order. Each row carries its
original `index`, captured `name`, fit estimates or a `failure` reason.
For Q3/Q1, the first row has `index: 2`, not index 0. Use that original index
in override paths. GE reports Primary `tones` and Post `post_tones` separately.

`writeback_applied` is true only when a completed receipt confirms applied items.
A failed receipt ends the recipe with the owner diagnostic; finally Save still runs.
It does not certify every fit or every saved format. `runs` identifies each tab,
adapter and acquisition, with its operation status, partial flag and Save
receipts. Full execution status also exposes owner conditions, summaries and
writeback candidates. Read the tab's fixed capture to inspect unmeasured tones;
the authored report does not recreate a separate complete-list `actual` table.

Field edits retain their supplied provenance. Expressions are labeled derived;
literal overrides are explicit. Retained Form values are not evidence of a
completed calibration.

The combined `measurement` plot labels curves by original index and source name
in participants order. Individual `measurement_i` and analysis `fit_i` plots use
the original index `i`. Pulse scans show each qubit's actual scan coordinates.
Delay scans show the common requested delay. GE shows probe-off/on shot traces.

A failed fit reports its reason; the owner summary retains quality diagnostics.
Nullable report estimates remain null when absent or nonfinite, never zero.
Full replies also contain the native analysis summary. Saved results keep the measured configuration and
names. Later Form edits, Reset or alias changes do not rewrite earlier reports.
Load Data changes the analysis source without changing current Form inputs.

## mux_twotone_spectrum

`mux_twotone_spectrum` opens `mux/twotone_spectrum`. It scans each participant's drive frequency and fits
its spectrum to estimate the qubit frequency and linewidth. Drives start
together at each scan point; mux readout waits for the longest drive.
Drive DAC frequencies advance in one shared hardware loop while mux readout
tones and ADC frequencies stay fixed.

Choose each `tones[i].drive`. Use dotted-path overrides to set MHz bounds at
`tones[i].sweep.start` and `stop`, or supply the full `sweeps` list.
Set shared `n` for all tones. The scans advance together by point number, not through every combination
of frequencies. Set `model_type` to `lor` or `sinc` to select the fit model.
Set `plot_fit` to false to hide the fitted curve without changing the fit.

Successful report rows contain `freq` and `fwhm` in MHz with their errors.
Inspect `measurement` and each `fit_i` for the peak and fit quality.
Check the fixed Run conditions for drive settings, sweep bounds, participants,
reps, rounds and recovery delay.

The recipe fits Primary, asks about Apply, then saves raw data and completed artifacts.
Candidates update each successful qubit's `freq` and `kappa` and add a frequency
sample at the captured bias. Review the destination, peak, linewidth and errors
using the [Save and Apply procedure](#save-review-and-apply).
[The frequency guide](../v2/mux/twotone_spectrum/guide.md) describes the GUI controls.

## mux_amp_rabi

`mux_amp_rabi` opens `mux/amp_rabi`. It scans drive gain on selected qubits and
fits Rabi oscillations to estimate pi and pi/2 gains.

Choose each `tones[i].drive`. Set `tones[i].sweep.start` and `stop` in native
gain units, or supply the full `sweeps` list and one shared `n`.
Copy a referenced drive inline before editing its fields.
Each scan point applies the corresponding
gain to all participants before mux readout.

Successful report rows contain `pi_gain` and `pi2_gain` with errors.
Inspect the oscillations in `measurement` and the fitted curves in `fit_i`.
Check drive settings and gain-sweep bounds in the fixed Run conditions.

The recipe fits Primary and asks about pulse updates. It saves raw data and completed artifacts after the decision.
Candidates copy the measured pulse into `<qubit>.pulses.pi_amp` and
`<qubit>.pulses.pi2_amp` with the fitted gains. Separate items update the x180/x90
module slots and `ext.pi_product`. Review pulse shape, gain and each destination
before [Apply](#save-review-and-apply).
[The amplitude Rabi guide](../v2/mux/amp_rabi/guide.md) covers individual GUI items.

## mux_len_rabi

`mux_len_rabi` opens `mux/len_rabi`. It scans drive duration and fits Rabi
oscillations to estimate pi and pi/2 lengths and the Rabi frequency.

Set `tones[i].drive` to a const or flat_top pulse. A flat_top duration must exceed
both ramps. Arb waveforms are not supported for this scan. Set each
`tones[i].sweep.start` and `stop` in us, or supply the full `sweeps` list.
Set shared `n` for all tones.
The recipe does not stretch or crop a waveform to make it valid.

Successful report rows contain `pi_len` and `pi2_len` in us and `rabi_f`
in MHz, with errors. Inspect `measurement` and `fit_i` for the oscillation
period and fit quality. Check drive settings and length-sweep bounds in the
fixed Run conditions.

The recipe fits Primary, asks about Apply, then saves raw data and completed artifacts.
Candidates include `ext.rabi_freq`, copied pi/pi2 pulses with fitted lengths,
x180/x90 module slots and `ext.pi_product`. Check the flat_top ramps and all pulse
and slot destinations in the [review step](#save-review-and-apply).
[The length Rabi guide](../v2/mux/len_rabi/guide.md) describes waveform restrictions.

## mux_t1

`mux_t1` opens `mux/t1`. It excites selected qubits, waits for a common delay,
and measures mux IQ to fit their energy relaxation.

Select calibrated pi pulses at `tones[i].drive`. Set the shared us range with
`"delays.start"`, `"delays.stop"` and `"delays.n"` overrides.
There is one requested-delay axis for all participants.

Successful report rows contain `t1` and optional second decay `t1b` in us
with errors. Compare the decay in `measurement` with each `fit_i`.
Check the shared delay range and pulse selections in the fixed Run conditions.

The recipe fits Primary and asks about T1 updates. It saves raw data and completed artifacts after the decision.
Candidates update the captured qubit's `t1` and add a T1 sample at captured bias.
Review the decay, error and destination before [Apply](#save-review-and-apply).
[The T1 guide](../v2/mux/t1/guide.md) describes the shared-delay measurement.

## mux_t2ramsey

`mux_t2ramsey` opens `mux/t2ramsey`. It applies x90, a common free-evolution delay
and a final x90, then fits decay or Ramsey fringes to estimate T2*.

Select x90 pulses at `tones[i].drive` and set each tone's `detune` in MHz.
Set the common us range with `"delay.start"`, `"delay.stop"` and
`"delay.n"`. The final x90 phase varies with detune and delay.

Successful report rows contain `t2` in us and fitted `detune` in MHz with
errors. Inspect `measurement` and `fit_i` for the fitted model, fringe rate and
decay. Check nominal detune, pulse selections and the shared delay range in the
fixed Run conditions.

The recipe fits Primary, asks about Apply, then saves raw data and completed artifacts.
Candidates update `t2r` and its sample at captured bias. A frequency correction
requires a fringe fit, finite source carrier, finite actual detune and finite
fitted fringe rate. Its value is source carrier + actual detune - fitted fringe
rate. Unavailable actual detune omits that correction but can leave a usable T2*
result. Check the phase-wrap convention in the
[Ramsey guide](../v2/mux/t2ramsey/guide.md#measurement-and-analysis) before
accepting a correction. Follow the [review procedure](#save-review-and-apply).

## mux_t2echo

`mux_t2echo` opens `mux/t2echo`. It applies x90, half the requested delay, x180,
the other half-delay and a final x90. It fits the resulting decay or fringes to
estimate echo coherence.

Select `tones[i].drive` for x90 and `tones[i].pi_pulse` for x180. Both pulses
for one qubit use the same generator. Set each tone's `detune` in MHz and the
shared us range at `"delay.start"`, `"delay.stop"` and `"delay.n"`.

Successful report rows contain `t2` in us and fitted `detune` in MHz with
errors. Inspect `measurement` and `fit_i` for the decay model and fit quality.
Check both pulse selections, nominal detune and shared delay range in the fixed
Run conditions.

The recipe fits Primary, then asks about `t2e` and a T2E sample at captured bias.
It saves raw data and completed artifacts after the decision. Echo does not propose a frequency correction.
Review those coherence results before [Apply](#save-review-and-apply).
[The Echo guide](../v2/mux/t2echo/guide.md) explains the two half-delays.

## mux_singleshot_ge

`mux_singleshot_ge` opens `mux/singleshot_ge`. It acquires paired probe-off/on
IQ shots for selected qubits. Primary fits their GE separation. Post uses that
Primary result to estimate a discrimination radius and confusion matrix.

Select each probe pulse at `tones[i].drive` and set `"shots"` to the number of
shots per preparation. An optional init pulse runs in both preparations.
The Form fixes reps and rounds at one. Shot pairing across tones remains intact.

Primary report rows contain `fidelity`, `theta`, `threshold` and `ge_s`.
Post rows contain `ge_radius`. Inspect the probe-off/on traces in `measurement`,
Primary `fit_i` plots and Post `post_i` diagnostics. Check shot count and pulse
selections in the fixed Run conditions.

The recipe fits Primary and Post before asking about Apply. It then saves raw shots and completed artifacts.
Primary candidates update `ext.readout_cal.fidelity`, `sigma`, `g_center` and
`e_center`. Post candidates update `radius` and the per-qubit 3x3
`confusion_matrix`. This is not a four-state joint assignment calibration.
Review cluster separation and both analysis stages before
[Apply](#save-review-and-apply).
[The GE guide](../v2/mux/singleshot_ge/guide.md) explains the diagnostics.

## Save, review and Apply

After Run, each recipe analyzes Primary and asks for a writeback decision.
GE also analyzes Post before asking. On acceptance it applies the selected
captured items. A single synchronous Save then writes raw data and all completed
artifacts before returning. The Run's `finally` also saves available output if
analysis or review raises. Analyze does not save automatically.
Inspect each output's save status and error in `runs`. Recipe completion alone does not
mean every requested output format saved successfully. Native `data.h5` is the
source for Load Data. Ramsey and Echo exports keep IQ on the delay/tone grid and
actual detune and source frequency on the tone-only grid in both native and
Labber files.

Review each candidate's original index, qubit name, value, error, fit quality and
destination. Destinations come from that Run or Load's captured bindings.
Current aliases cannot replace missing historical bindings. Failed or partially
acquired rows do not produce candidates.

The pending question includes its request id and captured candidates. Answer
with that request id, `accepted: true` and the selected unchanged `items`, or
with `accepted: false` and an empty item list. The host rejects stale request
ids and changed, unknown or duplicate candidates. Answering alone does not write.
The recipe explicitly applies accepted items, Primary then Post. Analysis,
preview and waiting for an answer do not write calibrations.

`cancel(execution=..., scope="run")` stops only acquisition and retains partial
data for subsequent analysis and Save. With no Run in progress it is a no-op.
`scope="recipe"` ends the script as cancelled, with a null report and the run
receipts. The recipe's finalizer still saves available output after Run.
A failed step reports its error without retrying acquisition. Inspect saved data
and artifact statuses before starting another run.
