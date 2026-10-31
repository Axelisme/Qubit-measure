# Local Z heralded

**Readers:** operator, agent. Run herald-conditioned local-Z correction and inspect acceptance.

## Inputs and sequence

Choose `two_qubit/local_z_heralded`. This is a separate experiment from `two_qubit/local_z`, with its own saved cfg and Result. Bind f1, f2, coupler and mux; calibrate the same pulses and full mux readout described in the [ordinary guide](guide.md). Both role tones need a circle classifier with a GE Post radius before Run. Other tones need no classifier. Reset the Form after changing captured sources.

Every shot emits optional reset, herald mux readout, the original preparation/CZ/projection sequence, then terminal mux readout. Set reset=None to skip reset. Herald and terminal use the same mux object, tones, gain, integration and classifier settings. The wait after herald is readout.post_delay.

Acquire both F1/F2 targets with consecutive nonnegative gate_counts, at least three points. n=0 omits CZ. Terminal phase is atan2(sigma_Y,sigma_X); sigma=(N_e-N_g)/N_total includes other. Complete rows use unwrap and count-propagated weighted least squares with intercept. Correction remains minus slope, without rewrapping. Both corrections must be valid before proposing the unchanged `$coupler.ext.cz_phase` pair with component identities. Changed source identities reject writeback.

## Counts and plots

Saved sums is float64 `(2,2,Nn,3,3,3,3)`. Leading scan axes follow the ordinary experiment. The final four axes are herald F1, herald F2, terminal F1, terminal F2, each ordered g/e/other. All 81 classes persist as raw accumulated counts. Rounds accumulate at acquisition. There is no host round axis or per-shot IQ storage.

Analysis slices both herald (g,g) and (e,e). It reverses ee target sigma to account for the target starting in e. The other qubit is also in e, so the ee slope contains local Z plus the conditional phase per gate. It fits gg and ee independently, without a shared slope, and reports ee−gg for each target. Only the gg correction pair is proposed for writeback.

Each complete target row uses phase=atan2(sigma_Y,sigma_X), unwrap, then weighted least squares with intercept. For independent projections, phase variance follows the atan2 derivatives and each sigma variance is (P_e+P_g−sigma²)/N. Other stays in N. Missing phases, ambiguous adjacent half-turns or unavailable positive phase variances prevent that row's fit. Covariance is (XᵀWX)⁻¹. No uncertainty floor or purity correction is added.

The phase plot compares gg and ee contrasts, phases and fitted slopes. The quadratures plot shows both groups, with ee sign-corrected. The herald figure shows all nine herald fractions per scan row. Its (g,g) cell is pass fraction, not ground-state purity. Inspect accepted counts and fit reasons before applying calibration.

Target F1/F2 and quadrature are hardware loops. Only gate count stays on the host, since it changes the sequence structure. All groups retain the full saved axes.

Native Save retains all sums, axes and source cfg. Load leaves the Form unchanged. Analysis requires saved cfg and rejects old per-shot files without sums. Cancellation preserves completed rounds; unmeasured points remain NaN. Check the run stop/error state before interpreting partial results.

Preview every proposal before Apply Selected. Writeback keys and acceptance conditions match the ordinary version. Recalibrate after changing pulse or readout conditions. Herald selection does not establish gate fidelity.


## Notebook

Use the validated LocalZHeraldedCfg as `cfg`. Supply the notebook's existing handles and device manager. `point_view` selects the save destination.

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.local_z.core_heralded import LocalZHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(LocalZHeraldedExp())
record = exp.run(cfg)
analysis = exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```
