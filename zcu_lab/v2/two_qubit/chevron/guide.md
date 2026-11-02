# Rabi chevron

**Readers:** operator, agent. Run `two_qubit/chevron` and interpret its count-derived return.

## Sequence and calibration

Scan common drive frequency in MHz and common envelope duration in us.
Each shot emits optional reset, nominal B11 preparation, simultaneous two-line
drive, double Xπ contrast and one terminal mux readout.
The ordinary version retains every classified shot without herald selection.
`reset=None` emits no reset. Set a reset cfg only after reviewing its pulse
and timing against the authorized limits. The Form's Reset action recaptures
inputs; it does not emit a reset pulse.

Both role tones require circle classifiers with g/e centers and radius.
Run mux singleshot GE Post under these readout conditions before Run.
Missing classifiers, linear classifiers and missing radii make the cfg invalid.
Classification happens during acquisition, not during analysis.
The three classes are g, e and other. Other stays in the denominator.

## Prepare the Form

Bind `f1`, `f2`, `coupler` and `mux`, then Open or Reset.
Review the captured identities and original tone positions from the mux group.
Check both Xπ pulses and all mux tones, ADC routes and scalar timing.
Readout references `$mux.module.readout` when present. Otherwise fill the blank
mux template, including classifiers. Missing sources remain required inputs.

The coupler's `cz_f1` and `cz_f2` slots seed editable ordinary charge pulses.
Without slots, fill both cosine templates, including signed gain, frequency,
envelope length and Nyquist zone. Channels come from captured role wiring.
Only F1/F2 charge lines play the drive; the coupler stays DC-biased.
Both source frequencies, envelope lengths and pre-delays must match.
The sequence waits for both drive pulses, including their individual post-delays.
Keep the signed gain ratio fixed.
At each frequency coordinate, both frequencies change together.
Individual phases remain fixed; choose relative phase with constructive_phase first.
Only common envelope duration changes within each trace.

Enter finite scan endpoints and strictly positive durations.
Review repetitions, rounds, optional reset and timing before Run.
Refresh updates choices without overwriting authored inputs.
After changing the parameter source, use the Form's Reset to recapture inputs.
Reset can overwrite unsaved inputs.

## Read the figures

Analyze publishes these figures and returns `None`. Per-frequency rates, counts
and curves stay inside analysis; frequency and duration selection remain manual.

The live traces and chevron map show P00 after double Xπ contrast.
P00 is the terminal (g,g) count divided by all nine terminal class counts.
The return figure shows all nine g/e/other joint outcomes.
These figures include preparation and readout errors.
They do not measure gate fidelity or resolve physical leakage by themselves.
There are no mean-IQ diagnostics or saved IQ clouds.

Each fully acquired duration trace with nonzero selected counts uses an
undamped cosine fit.
Rates below 1 divided by the saved duration span are unresolved.
Failed or degenerate fits report a reason rather than a zero rate.
Compare the measured return traces with the fitted curves.
Choose spacing so the expected rate is below `1/(2 × duration spacing)`.
Check aliasing manually. Thermal mixtures may contain more than one rate.

Frequency and duration selection remain manual. A valley only identifies a
window for later amplitude and phase calibration.

## Save and load

Native save keeps the complete source cfg, coordinate and duration axes, and
float64 `sums` with shape `(Nc,Nd,3,3)`.
Class axes are terminal F1, terminal F2.
Every class axis is ordered g, e, other.
Rounds accumulate during acquisition without normalization.
File size does not grow with repetitions or rounds.
Completed rounds survive cancellation; unmeasured points remain NaN.
Partial rounds remain visible.
Fits require all requested shots at every point of a duration trace.
Always check the execution status before accepting partial results.

Load leaves the current Form unchanged. Analysis uses the saved cfg and counts.
Old raw-IQ files without `sums` are rejected; there is no format fallback.

Use `two_qubit/chevron_heralded` for the version conditioned on a ground-ground herald.
After amplitude/frequency selection, measure conditional phase and local Z.
Repeat calibration after changing pulse shapes or the pair's working point.
Validate the actual FTF control chain separately from offline compile checks.

## Notebook

Use the adapter with a complete saved or operator-reviewed cfg.
`point_view` is the bound PointView for the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.chevron.core import ChevronExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(ChevronExp())
record = exp.run(cfg)
exp.analyze(None)
exp.save_run(point_view, source=record)
```

The coordinate is a hardware sweep. Constant and flat-top pulse durations also
use a hardware sweep in the same program. Sampled cosine, Gaussian and DRAG
waveforms reject swept lengths, so only their duration uses a host loop.

Fits use count-weighted binomial errors. Endpoint probabilities use
Jeffreys-smoothed variance so observations at zero or one retain finite weight.
