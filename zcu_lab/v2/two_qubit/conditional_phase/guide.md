# Conditional phase

**Readers:** operator, agent. Measure classified conditional phase and review a common CZ frequency proposal.

## Prepare the inputs

Both Forms show reset, Xπ, X90, dual-line drive and mux readout first.
Set the target and the outer `gate_count` before the inner
`frequency` sweep, then the acquisition settings.

Bind `qubit1` and `qubit2` to distinct qubits. Bind `mux` to a group containing both, and `coupler` to the component that stores their CZ pulses.
Notebook uses the same built-in position names, but does not bind the GUI aliases.
For an existing point, bind the new aliases. Rename both CZ slots and their library keys to
`cz_qubit1` and `cz_qubit2`. Make each slot point to its renamed key.

Calibrate Xπ, X90 and the complete mux readout first. Each qubit tone needs a circle classifier with radius from mux singleshot GE Post. Missing calibration, linear classification or missing radius stops Run with a GE Post prompt. Other mux tones need no classifier.

Readout references `$mux.module.readout` when present. Without a slot, fill all required fields in the blank mux template. Opening the tab or choosing Defaults captures calibrated pulses and sets two hidden tone indices from `$mux.qubits`. Refresh keeps the draft and rechecks the [mux tone count](../guide.md#綁定位置與讀出-tones) against the new capture. Use Defaults to replace the draft with initial values from the new source, then check them again.

The Form copies `$coupler.module.cz_qubit1` and `cz_qubit2` into editable pulses. Missing CZ slots remain required. CZ, Xπ and X90 must use the generator for that qubit. The two CZ lines require different generators, equal finite frequency, equal envelope duration and equal pre-delay. Signed gains, relative phase, waveform and individual post-delays remain explicit. Playback starts both lines together and waits for both to finish. The coupler stays DC-biased.

Choose lowercase `target=qubit1` or `qubit2`. Enter a strictly increasing common CZ frequency range in MHz and a `gate_count` list of unique positive integers, initially `[1,2,3]`. Each point replaces both CZ frequencies and preserves gains, phase and waveform. Set reps, rounds and delays for the authorized budget. Rounds accumulate counts at acquisition, without a host round scan.

## Sequence and counts

`two_qubit/conditional_phase` emits optional reset, spectator I or Xπ, target X90, n simultaneous CZ pulse pairs, X90 or Y90 projection, then terminal mux readout. `reset=None` emits no reset. The spectator branches describe nominal preparations on the initial ensemble.

Spectator I/Xπ, quadrature and frequency are hardware loops. Only gate count stays on the host because it changes the sequence length. X and Y projections use independent shots. Preparation has phase φ. X projection uses φ; Y projection uses φ+90°. Terminal Qubit1/Qubit2 are classified during acquisition into g/e/other.

Saved `sums` has shape `(2,2,Nn,Nf,3,3)`, ordered spectator_states, projection, gate count, frequency, terminal Qubit1, terminal Qubit2. Class order is g/e/other. These float64 values are unnormalized totals, not probabilities or per-shot IQ. Unmeasured points remain NaN. Cancellation preserves completed rounds. Each X/Y projection must total reps × rounds from the saved cfg before it can supply phase or fit data. A finite partial-round total is not complete. Per-frequency fits remain available on the complete frequency prefix, but an incomplete frequency axis prevents crossing selection and writeback. StopSignal carries only a stop request and reason. Acquisition failures propagate from Run as exceptions. A normal return after a stop request may contain partial counts.

## Read phase and quadratures

Analyze returns only `frequency_mhz` and `frequency_reason`. Per-frequency phases,
slopes, crossings, counts and herald fractions stay inside analysis for plotting.

For each target projection, `sigma=(N_e-N_g)/N_total`. Other remains in the denominator. Contrast is `hypot(sigma_X,sigma_Y)`; phase is `atan2(sigma_Y,sigma_X)` in degrees, wrapped to `[-180,180)`. Analysis does not project mean IQ or reclassify shots.

Use the live X/Y panels to check contrast for the current CZ train as the
completed-point count advances. Each gate count has its own Analyze quadrature
figure and its own contrast/wrapped-phase figure. Read X and Y by the nominal
spectator preparation. Their bars are ±1 shot SEM from the classified
(e-g) indicator with other contributing zero. The denominator is the measured
count at that point. The Slopes figure shows the fitted phase per CZ and its
fit standard error, not shot SEM. Use the π crossing figure to choose the
common frequency.

The fit uses X=A cos(n·slope+φ) and Y=A sin(n·slope+φ), with one amplitude A and phase offset φ per spectator and frequency. There is no quadrature DC offset. Count-derived sigma variances weight the fit. Other counts remain in the denominator. Zero contrast has no diagnostic phase, and unmeasured points block the corresponding slope fit.

Use at least two distinct gate counts whose differences have gcd=1. The default is [1,2,3]. Odd-only counts such as [1,3,5] have a π slope alias and are rejected before Run. After zero acceptance or unavailable positive variances remove points, each effective fit group must still satisfy this rule. An aliased group retains its diagnostic plots and reason, but no fit or writeback. Slopes use a representative of their unavoidable 2π period.

## Select a frequency and apply

Conditional phase is the fitted per-gate slope_e−slope_g, not the total phase of a selected train. Analysis unwraps adjacent frequency points and interpolates its π modulo 2π crossings inside the scan. This assumes sufficiently fine frequency sampling. A missing slope, no crossing, multiple crossings, a flat π segment or an ambiguous adjacent half-turn prevents a proposal. An endpoint crossing counts once. The interpolation does not assign a frequency uncertainty.

FIT proposes source pulses at `$coupler.pulses.cz_qubit1` and `cz_qubit2`, replacing only their common frequency. Gains, phases, waveforms and timing stay unchanged. Separate `$coupler.module.cz_qubit1` and `cz_qubit2` slot rows point to these keys. Both destinations use the original Run or Load capture, even if you change aliases afterward. Writeback fails without that capture. Preview each row before Apply Selected. Rows apply independently, so partial selection can update one line only. Choose Defaults in the next form to capture accepted pulses.

Amplitude and local-Z calibration remain separate steps. Recalibrate after changing shape or operating point.

## Save and load

Native Save retains sums, scan coordinates and source cfg (schema 7.0). Load leaves the current Form unchanged.
Native Load rejects older results whose axis names do not match this layout. Keep the original file and use the matching older version to read it.
If the result layout matches but the cfg major does not, Load retains Result and metadata without cfg. Phase analysis requires cfg.

For herald conditioning, choose the separate `two_qubit/conditional_phase_heralded` entry and read its guide.


## Notebook

Use the validated ConditionalPhaseCfg as `cfg`. Supply the notebook's existing handles and device manager. `point_view` selects the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.conditional_phase.core import ConditionalPhaseExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(ConditionalPhaseExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```

## Calibration order and field units

Calibrate each qubit's Xπ and Xπ/2 slots with amplitude and length Rabi before using them.
Confirm pulse frequency in MHz, signed gain in [-1, 1], and duration in us.
Use lookback to set mux timing, then mux onetone to set frequencies and gains.
Confirm readout timing in us and gains in [-1, 1].
Locate the dual-line CZ pulse with conditional spectroscopy and chevron.
Confirm the coupler's `cz_qubit1` and `cz_qubit2` slots before copying them into the Form.
The operator must confirm the common CZ frequency range in MHz before Run.
