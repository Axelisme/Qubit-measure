# Mux two-tone spectrum

**Readers:** operator, agent. Run the mux qubit frequency sweep from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/twotone_spectrum` from **New Tab**. Without this group and alias, opening the
form reports an error.
The captured parameter source must contain every qubit and its matching resonator
in readout order. Readout references the module slot `$mux.module.readout` when
present, preserving the library's tones and scalar settings. Use a mux-family
module with the full group tone count. If the slot is empty, Readout starts
inline from the blank mux template with group calibration tones.
The generator comes from `$mux.wiring.gen_ch`, and tone i's ADC channel comes
from `$mux.wiring.ro_ch[i]`, in the group's resonator order. Each tone starts
with gain 0.1. These values are editable initial inputs, not verified hardware
settings. Complete the remaining required fields and confirm wiring, gain and
timing before running.

The Tones picker selects the element to edit. Participants selects the qubits to
run. Tone names, count and order stay fixed. Set Participants to a nonempty JSON
list of unique existing names, for example `["Q3", "Q1"]`. This runs original
positions 2/0 in that order, even while the picker shows Q2. Empty, duplicate or
unknown participants block Run and remain editable for correction.

Every tone needs valid drives, including nonparticipants. Each tone uses its own
drive generator, distinct from the readout generator. Errors in another picker element
appear on the Tones row. Select that element and correct it to enable Run.
Reps, rounds and recovery delay are shared. Recovery initially uses five times
the largest captured T1, with the default for missing T1 values. A stored invalid
T1 leaves recovery blank for correction.

## FTF cost example

A two-participant offline DAC sweep used 16 joint points, 300 reps and 1 round.
Q1 swept 596 to 602 MHz and Q2 swept 295 to 301 MHz.
Both drives were 1 us const pulses at gain 0.1.
Recovery was 1007.509 us; readout and ADC integration were 1 and 0.5 us.
The recorded Run action took 1.206 s, about 0.075 s per joint sweep point.
The count is 16, not 16 squared, because the DAC ranges advance together.

The timer includes inbox transport, ComponentDriver keyboard/mouse actions,
terminal waiting and screenshots. It is not acquisition-only time and excludes
Analyze and Save. Do not compare it with timings from other slots or experiments
as a speed test. This example uses the current single-program DAC sweep.
Historical host-side software-sweep timings do not estimate this path.

Frequency point count, drive duration and shape, participants, reset, recovery
and readout windows affect cost. Reps and rounds repeat hardware acquisitions.
FTF compilation and cached evolution can dominate, so wall time need not scale
with reps. Start with a short scan at the intended drive length.
This 16-point example had large linewidth uncertainty and no calibration Apply.

## Measurement and analysis

Each tone supplies a drive-frequency range in MHz.
All tones use the same point count, including nonparticipants.
At each point, the selected drives use their corresponding frequencies together.
The sequence is common reset, simultaneous drives, then mux readout after the
longest drive. Pulse pre/post delays stay as configured.
One compiled program advances all drive DAC frequencies in the same hardware
loop. It does not scan combinations of the independent ranges. Mux readout DAC
tones and ADC frequencies stay fixed; this scan does not sweep readout frequency.
The saved axes use each drive's hardware-rounded start and step, which can differ
from rounding each requested point independently.

Choose the Lorentzian or sinc model. Each tone needs at least six valid points.
Frequency, FWHM and errors are in MHz. Check model quality and uncertainty before
using a fit as calibration.

## Writeback

Each qualifying tone proposes its captured qubit's `freq` and `kappa` in MHz,
plus a frequency sample at the captured bias.
Destinations use the same Run/Load source's full qubit binding.
Review the original tone position and captured destination before applying.

## Results and saved sources

Run draws one `measurement_i` figure per participant, without a shared overlay.
Each figure uses that tone's hardware-rounded scan coordinates.
The index `i` is the original tone position in the saved configuration.
Analyze uses the selected Run/Load source. Use the Analysis figure picker for
`fit_i`.
Here `i` is the original tone position, so Q3/Q1 uses indices 2/0.
Summary rows retain that order and show values, quality or a failure reason.
A missing source configuration leaves the name unavailable.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or saved run and preserves current Form inputs.
The selected source fixes the historical pulse, binding and writeback destination.
Cancellation retains completed-round averages for the full sweep.
If no round completed, signals remain NaN.
Partial data from a failed operation retains its failure status.
Only successful, fully finite rows qualify for calibration candidates.
Missing captured bindings suppress candidates; inconsistent bindings report an error.
Read each Writeback destination, value, quality and uncertainty.
Deselect unwanted items, then use **Apply Selected** to write the selected items.
Analyze and preview only calculate results. Each candidate needs its own selection.

## Drafts and MCP inputs

Saved drafts retain invalid field text and Participants for correction.
Restoration reports dropped fields and keeps the original parameter source.
Unreadable saved inputs produce a diagnostic and fresh defaults.
Reset captures current parameters and rebuilds initial values.

MCP field paths use original tone positions, for example
`("tones", 0, "drive", "gain")`. The integer selects the full tone list.
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_twotone_spectrum) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_twotone_spectrum`.
