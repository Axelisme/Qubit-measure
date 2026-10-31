## Behavior

**Readers:** operator, agent. How to run single-shot length Rabi calibration from the GUI or MCP and interpret the results.

Single-shot Length Rabi sweeps the qubit-drive pulse length, preserves every raw IQ shot, and derives ground / excited / other population curves during live view and analysis. Runs on real hardware.

Const, flat_top and sampled waveforms such as cosine, gauss and DRAG use the same host-side scan. Each length compiles one program and acquires one cloud of Shots raw IQ values. The result has shape (number of lengths, Shots), without averaging across shots or software rounds. Completed length rows survive cancellation; unmeasured rows remain NaN. Analyze uses the existing joint Rabi and readout-distribution fit.

Native arb waveforms have an asset-derived length and raise ValueError before scanning. Choose a waveform with an editable length for this experiment.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1. Length sweep is 0.03 us to 4 * `$qubit.module.x180.waveform.length`.

Raw acquisition and joint fitting require one ADC: Pulse or a single-tone Mux. Live preview uses that readout's circle classifier under Readout → Advanced. Without a classifier, Run still collects raw IQ and Analyze still fits; only the population preview is skipped. A supplied linear classifier cannot show Other; a circle without radius prompts singleshot GE Post.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The length sweep runs from 0.03 us to 4*pi_len. A missing or null stored pi_len uses a 1 us stop, including for arbitrary waveforms whose scalar length is absent. An invalid stored pi_len leaves the required stop empty for correction. A valid pi_len is a finite positive number, with booleans treated as invalid.

An existing x180 stays selected through its module slot and keeps its gain. If x180 is absent, the qub_probe template starts at gain=1. The default stop depends only on the stored scalar length.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

The qubit pulse uses the x180 module when present and the qub_probe template when absent, with the starting values described above. Reset uses optional `$qubit.module.reset`; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

A backend-valid joint fit with finite centers, radius and confusion matrix proposes one complete readout-module replacement at `$resonator.module.readout`. Pulse updates `classifier`; a single-tone Mux updates `tones.0.classifier`. The proposal preserves library waveform keys and rejects differences in other tested settings. Missing saved cfg or parameter capture fails explicitly. Invalid or nonfinite fits produce no proposal.

## Recommended

Set Initial State to the predominant state before the swept drive pulse even when the first sweep point is nonzero. Fit phase offset allows a correction within +/-90 degrees of that initial-state direction; the extrapolated zero-length population can then differ from the pre-pulse population. It is disabled by default. A prior singleshot/ge calibration is optional for raw acquisition. A sweep spanning a few pi lengths captures a full oscillation. Review the measured population curves and overlaid joint-fit curves before applying the complete module proposal.

**Last updated:** 2026-11-01
