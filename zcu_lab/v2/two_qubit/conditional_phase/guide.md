# Conditional phase

**Readers:** operator, agent. Measure classified conditional phase and review a common CZ frequency proposal.

## Prepare the inputs

Bind `f1` and `f2` to distinct fluxoniums, `coupler` to their transmon, and `mux` to a MuxGroup containing both qubits. Calibrate Xπ, X90 and the complete mux readout first. Each role tone needs a circle classifier with radius from mux singleshot GE Post. Missing calibration, linear classification or missing radius stops Run with a GE Post prompt. Other mux tones need no classifier.

Readout references `$mux.module.readout` when present. Without a slot, fill all required fields in the blank mux template. Open or Reset captures identities, calibrated pulses and original tone positions. Refresh preserves authored inputs. After changing sources, use Reset and check the inputs again.

The Form copies `$coupler.module.cz_f1` and `cz_f2` into editable pulses. Missing CZ slots remain required. CZ, Xπ and X90 must use their captured role generator. The two CZ lines require different generators, equal finite frequency, equal envelope duration and equal pre-delay. Signed gains, relative phase, waveform and individual post-delays remain explicit. Playback starts both lines together and waits for both to finish. The coupler stays DC-biased.

Choose lowercase `target=f1` or `f2`. Enter a strictly increasing common CZ frequency range in MHz and unique positive integer `gate_counts`, initially `[1,3,5]`. Each point replaces both CZ frequencies and preserves gains, phase and waveform. Set reps, rounds and delays for the authorized budget. Rounds accumulate counts at acquisition, without a host round scan.

## Sequence and counts

`two_qubit/conditional_phase` emits optional reset, spectator I or Xπ, target X90, n simultaneous CZ pulse pairs, X90 or Y90 projection, then terminal mux readout. `reset=None` emits no reset. The spectator branches describe nominal preparations on the initial ensemble.

X and Y projections are independent acquisitions. Preparation has phase φ. X projection uses φ; Y projection uses φ+90°. Terminal F1/F2 are classified during acquisition into g/e/other.

Saved `sums` has shape `(2,2,Nn,Nf,3,3)`, ordered branch, projection, gate count, frequency, terminal F1, terminal F2. Class order is g/e/other. These float64 values are unnormalized totals, not probabilities or per-shot IQ. Unmeasured points remain NaN. Cancellation preserves completed rounds. Inspect the run stop/error state before treating partial data as successful.

## Read phase and quadratures

For each target projection, `sigma=(N_e-N_g)/N_total`. Other remains in the denominator. Contrast is `hypot(sigma_X,sigma_Y)`; phase is `atan2(sigma_Y,sigma_X)` in degrees, wrapped to `[-180,180)`. Analysis does not project mean IQ or reclassify shots.

The phase plot shows contrast, each branch's logical phase, and conditional phase `phase(Xπ)-phase(I)`. The quadratures plot shows target sigma for each branch, projection and gate count. Each phase uses all acquired counts in both projections, including partially completed round budgets. Empty or missing projections and exactly zero contrast give a reason rather than an angle. Weak nonzero contrast remains visible. Inspect it before interpreting phase.

Each train's phase is its total accrual. These are ensemble measurements with preparation and readout effects. Ground-state purity and gate fidelity require separate evidence.

## Select a frequency and apply

Only n=1 selects frequency. Analysis unwraps adjacent frequency points and finds π modulo 2π by piecewise-linear interpolation within the scan. This assumes adjacent true phase changes below 180° and a sufficiently fine grid. Missing points split interpolation intervals. No crossing, multiple crossings, flat π segments, ambiguous half-turn steps or missing n=1 prevent a proposal. An endpoint crossing counts once.

Higher-n crossings are diagnostics, not selection sources. This interpolation has no fitted frequency-uncertainty model. Review consistency manually.

FIT proposes source pulses at `$coupler.pulses.cz_f1` and `cz_f2`, replacing only their common frequency. Gains, phases, waveforms and timing stay unchanged. Separate `$coupler.module.cz_f1` and `cz_f2` slot rows point to these keys in the fixed capture. Preview each row before Apply Selected. Rows apply independently, so partial selection can update one line only. Reset the next Form to capture accepted pulses.

Amplitude and local-Z calibration remain separate steps. Recalibrate after changing shape or operating point.

## Save and load

Native Save retains sums, scan coordinates and source cfg. Load leaves the current Form unchanged. Analyze uses only saved cfg and counts. Missing historical cfg stops phase analysis. Old per-shot IQ files without sums are rejected; no compatibility conversion is provided.

For herald conditioning, choose the separate `two_qubit/conditional_phase_heralded` entry and read its guide.
