# Mux qubit recipes

**Readers:** operator, agent. Run the seven mux qubit recipes, save results and review calibration proposals.

These seven recipes measure selected qubits with simultaneous mux readout, save
raw data and fits, and ask before applying calibration updates.

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

All tones need valid drives, including qubits you do not measure. Drives use
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

All seven recipes accept the same optional inputs:

| Input | What to supply | If omitted or null |
| --- | --- | --- |
| `target` | MuxGroup component name, such as `MG3`, for this tab's `$mux` binding | Use global aliases |
| `reuse_tab_id` | An idle tab of the matching experiment, which the recipe resets | Open a new tab |
| `participants` | Nonempty, unique qubit names in measurement order | Keep Reset defaults |
| `library` | A list of `{path, key}` library selections | Keep current selections |
| `inline` | A list of `{path}` references to copy for local editing | Keep references |
| `overrides` | A list of `{path, value}` field edits | Keep Form values |

The recipe opens or resets the tab, selects library entries, copies references
inline, then applies participants and overrides in one batch. Each list runs in
the order supplied. A null library key retains and validates the current
selection. A null override value retains that input. Copy referenced parents
before their children, outermost first. A later error leaves completed edits in
the tab; it does not undo them. Invalid fields remain visible and block Run,
even on nonparticipants.

Paths are JSON lists. Strings name fields and integers select positions in the
complete tone list. For complete tones Q1/Q2/Q3, `participants: ["Q3", "Q1"]`
measures original positions `[2, 0]`. Q3 edits still use position 2, not row 0
or the string `Q3`.

This example selects a Q3 pulse, copies it inline and changes its gain:

```json
{
  "target": "MG3",
  "participants": ["Q3", "Q1"],
  "library": [{"path": ["tones", 2, "drive"], "key": "Q3.pulses.pi"}],
  "inline": [{"path": ["tones", 2, "drive"]}],
  "overrides": [{"path": ["tones", 2, "drive", "gain"], "value": 0.3}]
}
```

Use your entry's actual component names, library keys and approved gains.
`tab.describe_cfg` shows fields, positions, names and available commands.
`tab.read_elements` shows the complete tone list. Frequency inputs use MHz, time
inputs use us and gain inputs use native units. Sweep ranges have `start`, `stop`
and `n` fields. For example, `["tones", 2, "sweep", "n"]` identifies Q3's point
count.

An override edits only its path. Set individual range leaves rather than
expecting a container edit to merge with the previous value. An expression value
uses `{"expr": "..."}` and remains an expression in the Form. Reps, rounds,
recovery delay and the delay-scan axis are shared, not per-qubit controls.

## Read conditions and results

Each recipe returns `status`, an authored `report` and `runs`.
The report's `tones` list follows native analysis order. Each row carries its
original `index`, captured `name`, fit estimates or a `failure` reason.
For Q3/Q1, the first row has `index: 2`, not index 0. Use that original index
in override paths. GE reports Primary `tones` and Post `post_tones` separately.

`writeback_applied` records whether an accepted answer reached Apply.
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

Choose each `tones[i].drive`. Use typed overrides to set MHz bounds at
`tones[i].sweep.start` and `stop`, and the point count at `tones[i].sweep.n`. Every tone's sweep must have the same point
count. The scans advance together by point number, not through every combination
of frequencies.

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
gain units and the point count at `tones[i].sweep.n`. Copy a referenced drive inline before editing its fields.
All tone sweeps need equal point counts. Each scan point applies the corresponding
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
`tones[i].sweep.start` and `stop` in us, and `tones[i].sweep.n` to the point count.
All tones need equal point counts.
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
`["delays", "start"]`, `["delays", "stop"]` and `["delays", "n"]` overrides.
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
Set the common us range with `["delay", "start"]`, `["delay", "stop"]` and
`["delay", "n"]`. The final x90 phase varies with detune and delay.

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
shared us range at `["delay", "start"]`, `["delay", "stop"]` and `["delay", "n"]`.

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

Select each probe pulse at `tones[i].drive` and set `["shots"]` to the number of
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
`scope="script"` ends the script as cancelled, with a null report and the run
receipts. The recipe's finalizer still saves available output after Run.
A failed step reports its error without retrying acquisition. Inspect saved data
and artifact statuses before starting another run.
