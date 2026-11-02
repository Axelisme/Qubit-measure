# Constructive relative phase with heralding

**Readers:** operator, agent. Run `two_qubit/constructive_phase_heralded` and interpret its count-derived return and phase proposals.

## Sequence and calibration

Scan F2 minus F1 relative phase in degrees and common envelope duration in us.
Each shot emits optional reset, a full mux herald read, nominal B11 preparation,
simultaneous two-line drive, double Xπ contrast and terminal mux readout.
Herald and terminal use the same readout object and cfg.
The readout's post_delay sets the wait after herald.
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
Each phase coordinate sets F2.phase to saved F1.phase plus that coordinate.
The common frequency and F1 phase remain fixed. Choose the transition using
conditional spectroscopy and the duration window using chevron.
Only common envelope duration changes within each trace.

Enter finite scan endpoints and strictly positive durations.
Review repetitions, rounds, optional reset and timing before Run.
Refresh updates choices without overwriting authored inputs.
After changing the parameter source, use the Form's Reset to recapture inputs.
Reset can overwrite unsaved inputs.

## Read the figures

Analyze returns only `relative_phase_deg`, `reason` and `interference_quality`.
Per-phase rates, populations and interference curves stay inside analysis for plotting.

The live traces and interference map show P00 after double Xπ contrast.
Analysis first selects counts whose herald is classified as (g,g).
Terminal P00 is then its (g,g) count divided by the selected terminal total.
The separate herald figure shows all nine herald fractions at every scan point.
Its (g,g) cell is pass fraction, not ground-state purity.
Zero accepted counts produce NaN probabilities and unresolved fits.
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

Identifiable rates use `rate² = A + B cosφ + C sinφ`.
The proposed relative phase is `atan2(C,B)` in degrees, the maximum-rate branch.
More than three identifiable phase slices and full-rank phase coverage are
needed for finite uncertainty. Zero modulation or nonfinite uncertainty prevents
a proposal. This phase is not a logical local-Z correction.

## Save and load

Native save keeps the complete source cfg, coordinate and duration axes, and
float64 `sums` with shape `(Nc,Nd,3,3,3,3)`.
Class axes are herald F1, herald F2, terminal F1, terminal F2.
Every class axis is ordered g, e, other.
Rounds accumulate during acquisition without normalization.
File size does not grow with repetitions or rounds.
Completed rounds survive cancellation; unmeasured points remain NaN.
Partial rounds remain visible.
Fits require all requested shots at every point of a duration trace.
Always check the execution status before accepting partial results.

Load leaves the current Form unchanged. Analysis uses the saved cfg and counts.
Old raw-IQ files without `sums` are rejected; there is no format fallback.

## Review and apply phase proposals

FIT proposes only F2's ordinary charge pulse and its independent coupler slot.
The writeback targets remain `<coupler>.pulses.cz_f2` and
`<coupler>.module.cz_f2` in both versions.
F2.phase becomes saved F1.phase plus the fitted relative phase.
F2 frequency, signed gain, waveform and timing stay as measured.
F1 stays the phase reference. Missing cfg or an unresolved phase gives no proposal.
Inspect the captured destination and waveform before Apply.
The pulse and slot rows apply independently; review partial selections.
Only an applied proposal becomes calibration for later Forms.

Use `two_qubit/constructive_phase` for the ordinary scan without post-selection.
After amplitude/frequency selection, measure conditional phase and local Z.
Repeat calibration after changing pulse shapes or the pair's working point.
Validate the actual FTF control chain separately from offline compile checks.

## Notebook

Use the adapter with a complete saved or operator-reviewed cfg.
`point_view` is the bound PointView for the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.constructive_phase.core_heralded import ConstructivePhaseHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(ConstructivePhaseHeraldedExp())
record = exp.run(cfg)
exp.analyze(None)
exp.save_run(point_view, source=record)
```

## Excited-excited control

Analysis also selects herald (e,e). Preparation flips this state to |00⟩.
The final double Xπ should return it to ee. The ee control plot shows one
minus its terminal ee probability, including other outcomes in the denominator.
Zero accepted counts give NaN. This control has no fit shared with gg.
Only the gg slice supplies calibration proposals and writeback.

The coordinate is a hardware sweep. Constant and flat-top pulse durations also
use a hardware sweep in the same program. Sampled cosine, Gaussian and DRAG
waveforms reject swept lengths, so only their duration uses a host loop.

Fits use count-weighted binomial errors. Endpoint probabilities use
Jeffreys-smoothed variance so observations at zero or one retain finite weight.
