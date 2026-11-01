## Behavior

**Readers:** operator, agent. How to calibrate the single-shot AC-Stark response from the GUI or MCP and interpret the results.

Single-shot AC-Stark calibration: drives a cavity Stark tone and a qubit probe tone while sweeping cavity gain and qubit frequency (2D), classifying each shot during acquisition against the |g>/|e> IQ-cluster centres. Fits the Stark-shifted resonance versus gain² to extract the AC-Stark coefficient (photon number per gain²). Runs on real hardware.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1. Qubit sweep is `$qubit.freq` - 700 to `$qubit.freq` + 100 MHz. Qubit probe length/mixer use `$qubit.module.x180.waveform.length` and `$qubit.module.x180.mixer_freq`. Pre/post delay are 1.0/`$resonator.kappa` and 0.5/`$resonator.kappa`; cavity tone length adds 1.0/`$resonator.kappa` to the pi length.

Readout accepts only the pulse family.
A mux readout in the library is incompatible.
Classification uses the readout's circle classifier under Readout → Advanced.
Run requires a positive radius; missing radius prompts singleshot GE Post.
Linear has no Other state and is rejected.
Analyze corrects populations only with the saved readout cfg's confusion matrix.
Historical cfg without a classifier or matrix skips correction.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Reset and init start disabled. Cavity/qubit tones start from res_probe/pi_pulse with calibrated timing. Readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

Proposes the fitted AC-Stark coefficient at `$qubit.ext.ac_stark_coeff`, in photon number per gain². MIST uses it to rescale the x-axis to photon number.

## Recommended

Calibrate the readout with singleshot/ge Primary and Post; run after the dispersive-shift experiment has set '`$qubit.ext.chi`' / '`$resonator.kappa`'. Sweep the Stark gain across the onset and the probe frequency around the qubit line.

## Saved counts

Save keeps `counts` shaped `(Ngain, Nfreq, 3)` in `[g,e,other]` order, with
`population_states=[0,1,2]`. Other counts shots outside the accepted circles
or tied between them. It is not a calibrated leakage population.
Each point sums to the shots acquired across completed rounds. Live plots and
Analyze divide each point by its total before displaying populations or applying
the saved confusion matrix. Sweep coordinates and units are unchanged. Cfg schema
remains 3.0.

Old files storing `populations` instead of `counts` are unsupported. Load reports
the missing `counts` variable. Acquire a new run; Load does not convert stored
fractions to counts.

**Last updated:** 2026-10-29
