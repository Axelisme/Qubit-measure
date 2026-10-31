# Conditional phase heralded

**Readers:** operator, agent. Run herald-conditioned conditional phase and inspect acceptance.

## Inputs and sequence

Choose `two_qubit/conditional_phase_heralded`. This is a separate experiment from `two_qubit/conditional_phase`, with its own saved cfg and Result. Bind f1, f2, coupler and mux; calibrate the same pulses and full mux readout described in the [ordinary guide](guide.md). Both role tones need a circle classifier with a GE Post radius before Run. Other tones need no classifier. Reset the Form after changing captured sources.

Every shot emits optional reset, herald mux readout, the original preparation/CZ/projection sequence, then terminal mux readout. Set reset=None to skip reset. Herald and terminal use the same mux object, tones, gain, integration and classifier settings. The wait after herald is readout.post_delay.

Choose lowercase target f1/f2, positive unique gate_counts, default [1,2,3], with at least two points and difference gcd=1 and a strictly increasing common frequency scan in MHz. Terminal phase is atan2(sigma_Y,sigma_X); sigma=(N_e-N_g)/N_total includes other. Conditional phase comes from the shared per-gate slope_e−slope_g. A unique joint π crossing selects the writeback frequency. FIT proposes `$coupler.pulses.cz_f1` and `cz_f2` plus their `$coupler.module.cz_f1` and `cz_f2` slot rows. Review and apply rows separately.

## Counts and plots

Saved sums is float64 `(2,2,Nn,Nf,3,3,3,3)`. Leading scan axes follow the ordinary experiment. The final four axes are herald F1, herald F2, terminal F1, terminal F2, each ordered g/e/other. All 81 classes persist as raw accumulated counts. Rounds accumulate at acquisition. There is no host round axis or per-shot IQ storage.

Analysis uses both herald (g,g) and (e,e). In the ee group, target sigma reverses sign and nominal spectator I/Xπ means actual spectator e/g. Analysis reverses both sigma signs and reorders ee to actual g/e before fitting. It keeps separate amplitudes and phase offsets for each herald group while sharing the per-gate slope for each actual spectator and frequency. The model is X=A_h cos(n·slope+φ_h), Y=A_h sin(n·slope+φ_h), without quadrature DC.

Count-derived sigma variances weight both groups, with other in each denominator. Zero acceptance and unavailable positive variances remove points. Each group must retain at least two gate counts with difference gcd=1. Unmeasured axes or aliased groups retain diagnostic phases and a reason, but do not authorize joint writeback. Never combine two herald groups' gate-count sets to hide an alias. Odd-only counts such as [1,3,5] are rejected before Run.

The phase and quadratures plots show gg and sign-corrected ee by actual spectator. Slopes and conditional-phase plots compare the joint fit with the independent gg fit. Writeback uses only the joint slope_e−slope_g crossing. A missing, flat, ambiguous or nonunique crossing prevents a proposal; there is no gg fallback. Frequency interpolation supplies no uncertainty.

The herald figure shows all nine herald fractions for each scan row. Its (g,g) cell is pass fraction, not ground-state purity. A low pass fraction reduces accepted sample size. Inspect it alongside contrast and fit diagnostics before applying calibration.

Spectator, quadrature and frequency use hardware loops. Gate count stays on the host because it changes the sequence structure. Both read tags share one mux object within each freshly built program.

Native Save retains all sums, axes and source cfg. Load leaves the Form unchanged. Analysis requires saved cfg and rejects old per-shot files without sums. Cancellation preserves completed rounds; unmeasured points remain NaN. Check the run stop/error state before interpreting partial results.

Preview every proposal before Apply Selected. Writeback keys and acceptance conditions match the ordinary version. Recalibrate after changing pulse or readout conditions. Herald selection does not establish gate fidelity.


## Notebook

Use the validated ConditionalPhaseHeraldedCfg as `cfg`. Supply the notebook's existing handles and device manager. `point_view` selects the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.conditional_phase.core_heralded import ConditionalPhaseHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(ConditionalPhaseHeraldedExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```
