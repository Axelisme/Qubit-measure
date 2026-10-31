# Conditional phase

**Readers:** operator, agent. Measure classified conditional phase and review a common CZ frequency proposal.

## Prepare the inputs

Bind `f1` and `f2` to distinct fluxoniums, `coupler` to their transmon, and `mux` to a MuxGroup containing both qubits. Calibrate Xπ, X90 and the complete mux readout first. Each role tone needs a circle classifier with radius from mux singleshot GE Post. Missing calibration, linear classification or missing radius stops Run with a GE Post prompt. Other mux tones need no classifier.

Readout references `$mux.module.readout` when present. Without a slot, fill all required fields in the blank mux template. Open or Reset captures identities, calibrated pulses and original tone positions. Refresh preserves authored inputs. After changing sources, use Reset and check the inputs again.

The Form copies `$coupler.module.cz_f1` and `cz_f2` into editable pulses. Missing CZ slots remain required. CZ, Xπ and X90 must use their captured role generator. The two CZ lines require different generators, equal finite frequency, equal envelope duration and equal pre-delay. Signed gains, relative phase, waveform and individual post-delays remain explicit. Playback starts both lines together and waits for both to finish. The coupler stays DC-biased.

Choose lowercase `target=f1` or `f2`. Enter a strictly increasing common CZ frequency range in MHz and unique positive integer `gate_counts`, initially `[1,2,3]`. Each point replaces both CZ frequencies and preserves gains, phase and waveform. Set reps, rounds and delays for the authorized budget. Rounds accumulate counts at acquisition, without a host round scan.

## Sequence and counts

`two_qubit/conditional_phase` emits optional reset, spectator I or Xπ, target X90, n simultaneous CZ pulse pairs, X90 or Y90 projection, then terminal mux readout. `reset=None` emits no reset. The spectator branches describe nominal preparations on the initial ensemble.

Spectator I/Xπ, quadrature and frequency are hardware loops. Only gate count stays on the host because it changes the sequence length. X and Y projections use independent shots. Preparation has phase φ. X projection uses φ; Y projection uses φ+90°. Terminal F1/F2 are classified during acquisition into g/e/other.

Saved `sums` has shape `(2,2,Nn,Nf,3,3)`, ordered spectator_states, projection, gate count, frequency, terminal F1, terminal F2. Class order is g/e/other. These float64 values are unnormalized totals, not probabilities or per-shot IQ. Unmeasured points remain NaN. Cancellation preserves completed rounds. Inspect the run stop/error state before treating partial data as successful.

## Read phase and quadratures

For each target projection, `sigma=(N_e-N_g)/N_total`. Other remains in the denominator. Contrast is `hypot(sigma_X,sigma_Y)`; phase is `atan2(sigma_Y,sigma_X)` in degrees, wrapped to `[-180,180)`. Analysis does not project mean IQ or reclassify shots.

The phase plot shows contrast and logical phase for each spectator state and gate count. The quadratures plot shows classified X/Y sigma. The slopes plot shows the fitted phase per gate for actual spectator g and e.

The fit uses X=A cos(n·slope+φ) and Y=A sin(n·slope+φ), with one amplitude A and phase offset φ per spectator and frequency. There is no quadrature DC offset. Count-derived sigma variances weight the fit. Other counts remain in the denominator. Zero contrast has no diagnostic phase, and unmeasured points block the corresponding slope fit.

Use at least two distinct gate counts whose differences have gcd=1. The default is [1,2,3]. Odd-only counts such as [1,3,5] have a π slope alias and are rejected before Run. After zero acceptance or unavailable positive variances remove points, each effective fit group must still satisfy this rule. An aliased group retains its diagnostic plots and reason, but no fit or writeback. Slopes use a representative of their unavoidable 2π period.

## Select a frequency and apply

Conditional phase is the fitted per-gate slope_e−slope_g, not the total phase of a selected train. Analysis unwraps adjacent frequency points and interpolates its π modulo 2π crossings inside the scan. This assumes sufficiently fine frequency sampling. A missing slope, no crossing, multiple crossings, a flat π segment or an ambiguous adjacent half-turn prevents a proposal. An endpoint crossing counts once. The interpolation does not assign a frequency uncertainty.

FIT proposes source pulses at `$coupler.pulses.cz_f1` and `cz_f2`, replacing only their common frequency. Gains, phases, waveforms and timing stay unchanged. Separate `$coupler.module.cz_f1` and `cz_f2` slot rows point to these keys in the fixed capture. Preview each row before Apply Selected. Rows apply independently, so partial selection can update one line only. Reset the next Form to capture accepted pulses.

Amplitude and local-Z calibration remain separate steps. Recalibrate after changing shape or operating point.

## Save and load

Native Save retains sums, scan coordinates and source cfg. Load leaves the current Form unchanged. Analyze uses only saved cfg and counts. Missing historical cfg stops phase analysis. Old per-shot IQ files without sums are rejected; no compatibility conversion is provided.

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
