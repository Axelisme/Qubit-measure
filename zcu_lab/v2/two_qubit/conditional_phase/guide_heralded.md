# Conditional phase heralded

**Readers:** operator, agent. Run herald-conditioned conditional phase and inspect acceptance.

## Inputs and sequence

Choose `two_qubit/conditional_phase_heralded`. This is a separate experiment from `two_qubit/conditional_phase`, with its own saved cfg and Result. Bind f1, f2, coupler and mux; calibrate the same pulses and full mux readout described in the [ordinary guide](guide.md). Both role tones need a circle classifier with a GE Post radius before Run. Other tones need no classifier. Reset the Form after changing captured sources.

Every shot emits optional reset, herald mux readout, the original preparation/CZ/projection sequence, then terminal mux readout. Set reset=None to skip reset. Herald and terminal use the same mux object, tones, gain, integration and classifier settings. The wait after herald is readout.post_delay.

Choose lowercase target f1/f2, positive unique gate_counts and a strictly increasing common frequency scan in MHz. Terminal phase is atan2(sigma_Y,sigma_X); sigma=(N_e-N_g)/N_total includes other. Conditional phase, n=1 crossing selection and frequency writeback are unchanged. FIT proposes `$coupler.pulses.cz_f1` and `cz_f2` plus their `$coupler.module.cz_f1` and `cz_f2` slot rows. Review and apply rows separately.

## Counts and plots

Saved sums is float64 `(2,2,Nn,Nf,3,3,3,3)`. Leading scan axes follow the ordinary experiment. The final four axes are herald F1, herald F2, terminal F1, terminal F2, each ordered g/e/other. All 81 classes persist as raw accumulated counts. Rounds accumulate at acquisition. There is no host round axis or per-shot IQ storage.

Analysis selects the terminal 3×3 table where herald classified both qubits g, then uses the ordinary classified phase analysis. It does not apply a purity correction. Missing or empty accepted projections and zero contrast retain reasons, not invented angles. Completed counts remain usable even if cancellation leaves fewer rounds than requested.

The phase and quadratures figures use accepted terminal counts. The herald figure shows all nine herald fractions for each scan row. Its (g,g) cell is pass fraction. Pass fraction measures classified acceptance, not ground-state purity. A low pass fraction reduces accepted sample size; inspect it alongside contrast before applying calibration.

Native Save retains all sums, axes and source cfg. Load leaves the Form unchanged. Analysis requires saved cfg and rejects old per-shot files without sums. Cancellation preserves completed rounds; unmeasured points remain NaN. Check the run stop/error state before interpreting partial results.

Preview every proposal before Apply Selected. Writeback keys and acceptance conditions match the ordinary version. Recalibrate after changing pulse or readout conditions. Herald selection does not establish gate fidelity.
