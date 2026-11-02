# Heralded CZ common-amplitude calibration

**Readers:** operator, agent. Run the heralded CZ amplitude scan and interpret accepted return counts.

## Inputs and sequence

Open `two_qubit/cz_amplitude_heralded`.
Use the calibrated roles, two-line drive and common scale range described in
the [ordinary guide](guide.md). This version has its own Form, cfg and saved Result.
F1/F2 both require circle classifiers with radius from GE Post.
Missing or linear calibration, including a missing radius, stops Run.

Each shot emits
`optional reset → herald mux → B11 → simultaneous CZ → Xπ⊗Xπ → terminal mux`.
Leave `reset` null to emit no reset.
Herald and terminal emit the same mux module with the same settings.
Readout post-delay supplies the wait after herald. There is no separate herald gain
or integration length.

## Accepted return and pass fraction

The live curve selects the herald g,g slice and displays its terminal P00.
FIT uses that same slice and the ordinary first-cycle fit.
P00 divides accepted terminal g,g counts by all nine accepted terminal classes,
including other. Empty accepted slices produce NaN, not a fabricated probability.

The herald figure displays nine herald probabilities against scale.
The g,g probability is the pass fraction, the fraction of acquired shots accepted
by both ground classifiers. It is not ground-state purity.
Classification errors and state changes between readouts still affect the return.

The fit requires all requested shots at each scale in its contiguous measured
prefix, counting every herald class when checking acquisition completeness.
It also needs nonzero accepted counts. Rejected shots do not make a fully
acquired scale incomplete. The remaining fit, uncertainty and proposal policy
matches the ordinary version.

## Writeback and saved counts

Both versions propose `$coupler.pulses.cz_f1`, `$coupler.module.cz_f1`,
`$coupler.pulses.cz_f2` and `$coupler.module.cz_f2`.
The accepted-count fit supplies the common gain multiplier.
Preview and Apply Selected use the saved cfg and matching Run/Load capture.
The experiment does not write parameters during Run or FIT.

Save stores float64 `sums` with shape `Ns × 3 × 3 × 3 × 3`.
The class axes are herald F1, herald F2, terminal F1 and terminal F2.
Each class axis is g, e, other.
All 81 count categories remain stored, including rejected herald classes.
Rounds add at acquisition; no per-shot IQ or mean-IQ diagnostic remains.
Unmeasured scales are NaN. Cancellation retains completed rounds.

Native Load rejects old per-shot files without sums.
The cfg schema is 2.0. Older cfgs using `scales` do not restore.
Their saved counts and axes still load and can be plotted, but fitting requires cfg.
Full run, saving and offline analysis use the saved cfg rather than current
Form calibrations. Missing saved cfg stops the fit.

## Notebook

Use the adapter with a complete saved or operator-reviewed cfg.
`point_view` is the bound PointView for the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.cz_amplitude.core_heralded import CZAmplitudeHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(CZAmplitudeHeraldedExp())
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

The common gain scale is a hardware sweep in one compiled program.

Fits use count-weighted binomial errors. Endpoint probabilities use
Jeffreys-smoothed variance so observations at zero or one retain finite weight.
