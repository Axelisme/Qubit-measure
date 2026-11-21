# Heralded Conditional phase

**Readers:** operator, agent. Run herald-conditioned conditional phase and inspect acceptance.

## Inputs and sequence

Choose `two_qubit/conditional_phase_heralded`. This experiment has its own saved cfg and Result. Bind `qubit1`, `qubit2`, `coupler` and `mux` as in the [ordinary guide](guide.md#prepare-the-inputs), which also explains how to rebind an existing point.
Notebook uses the same built-in position names, but does not bind the GUI aliases. Opening the tab or choosing Defaults sets two hidden tone indices from `$mux.qubits`.
Both qubit tones need a circle classifier with a GE Post radius before Run. Other tones need no classifier. Choose Defaults after changing captured sources.

Every shot emits optional reset, herald mux readout, the original preparation/CZ/projection sequence, then terminal mux readout. Set reset=None to skip reset. Herald and terminal use the same mux object, tones, gain, integration and classifier settings.

Choose `target=qubit1` or `target=qubit2` and a list of positive unique `gate_count` values, default [1,2,3], with at least two points and difference gcd=1 and a strictly increasing common frequency scan in MHz. Terminal phase is atan2(sigma_Y,sigma_X); sigma=(N_e-N_g)/N_total includes other. Conditional phase comes from the shared per-gate slope_e−slope_g. A unique joint π crossing selects the writeback frequency. FIT proposes `$coupler.pulses.cz_qubit1` and `cz_qubit2` plus their `$coupler.module.cz_qubit1` and `cz_qubit2` slot rows. Review and apply rows separately.

## Wait after herald

Set the readout's `post_delay` before scanning so photons in the resonator can
decay before the qubit drive. Herald and terminal readout share this value
([ADR-0074](../../../../docs/adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)).
The wait starts when the readout generator waveform ends, without waiting for
the ADC integration window to end
([ADR-0083](../../../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)).
The `mux/onetone` Form seeds this field from the linewidth. FIT does not
calibrate the wait.

If the scan has no π crossing, check `post_delay`. Photons left by the
herald readout can shift the qubit transition frequency and cause dephasing.
A nominal Xπ after herald may barely flip the qubit.

Use each resonator's `kappa` from the `mux/onetone` FIT. It is the full
width at half maximum (FWHM) in MHz. Start at `5 / (2π · κ)` µs, five photon
decay times and the same formula used for the ring-up seed. For a shared
`post_delay`, use the smallest κ so the slowest resonator has time to empty.

In a low-SNR FTF example, R2's linewidth was about 1.2 MHz. Five photon
decay times are about 0.66 µs, and 2 µs is about 15 times one photon decay
time. After gg herald and a nominal Qubit2 Xπ, Qubit2's terminal e fraction was
11.6% at 0 µs and 92.9% at 2 µs. These fractions do not subtract the I
baseline. The 2 µs value was a comparison point, not a minimum or an optimum.

In `two_qubit/joint_readout`, choose `reset/mux_herald` in Reset. Set
`keep` to g for both role tones, keyed by their positions in the full readout.
Use the same full mux readout for herald and terminal. After Run, inspect the
4×9 row-normalized probability matrix in FIT. Find the row with nominal Xπ
on the target and I on the other qubit.
Sum the columns where the target's terminal class is e. Compare with the
gg row for an I baseline if needed. This is the terminal e fraction among
shots that passed gg herald, not gate fidelity. Lengthen only `post_delay`
and repeat Run and FIT. If the longer wait still raises the terminal e
fraction, adjust `post_delay` before scanning for the π crossing.

## Counts and plots

Analyze returns only `frequency_mhz` and `frequency_reason`. Per-frequency phases,
slopes, crossings, counts and herald fractions stay inside analysis for plotting.

Saved sums is float64 `(2,2,Nn,Nf,3,3,3,3)`. Leading scan axes follow the ordinary experiment. The final four axes are herald Qubit1, herald Qubit2, terminal Qubit1, terminal Qubit2, each ordered g/e/other. All 81 classes persist as raw accumulated counts. Rounds accumulate at acquisition. There is no host round axis or per-shot IQ storage.

Analysis uses both herald (g,g) and (e,e). In the ee group, target sigma reverses sign and nominal spectator I/Xπ means actual spectator e/g. Analysis reverses both sigma signs and reorders ee to actual g/e before fitting. It keeps separate amplitudes and phase offsets for each herald group while sharing the per-gate slope for each actual spectator and frequency. The model is X=A_h cos(n·slope+φ_h), Y=A_h sin(n·slope+φ_h), without quadrature DC.

Count-derived sigma variances weight both groups, with other in each denominator. Zero acceptance and unavailable positive variances remove points. Each group must retain at least two gate counts with difference gcd=1. Unmeasured axes or aliased groups retain diagnostic phases and a reason, but do not authorize joint writeback. Never combine two herald groups' gate-count sets to hide an alias. Odd-only counts such as [1,3,5] are rejected before Run.

The live quadratures show herald (g,g) for the current CZ train and identify
the actual spectator state. Analyze splits (g,g) and sign-corrected (e,e)
quadratures into separate figures for each gate count. The X/Y bars are ±1
shot SEM of the classified target (e-g) indicator. Only shots in the
displayed herald group contribute to its SEM; neither other herald classes
nor attempted shots enter that denominator. Other terminal outcomes within
the displayed herald group contribute zero to the indicator. Negating the
(e,e) signal does not change its SEM. Contrast and wrapped phase have a
separate figure per gate count. Slopes show the joint and independent (g,g)
fit estimates with fit standard errors, not shot SEM. Use the π crossing
figure to compare the fits. Writeback uses only the joint slope_e−slope_g crossing. A missing, flat, ambiguous or nonunique crossing prevents a proposal; there is no gg fallback. Frequency interpolation supplies no uncertainty.

The herald figure shows all nine herald fractions for each scan row. Its (g,g) cell is pass fraction, not ground-state purity. A low pass fraction reduces accepted sample size. Inspect it alongside contrast and fit diagnostics before applying calibration.

Spectator, quadrature and frequency use hardware loops. Gate count stays on the host because it changes the sequence structure. Both read tags share one mux object within each freshly built program.

Native Save retains all sums, axes and source cfg (schema 4.0). Load leaves the Form unchanged.
Native Load rejects older results whose axis names do not match this layout. Keep the original file and use the matching older version to read it.
If the result layout matches but the cfg major does not, Load retains Result and metadata without cfg. Phase analysis requires cfg.

Cancellation preserves completed rounds; unmeasured points remain NaN. StopSignal carries only a stop request and reason. Acquisition failures propagate from Run as exceptions. A normal return after a stop request may contain partial counts. Phase and fits require every planned round of both projections, checked against reps × rounds using all 81 classes before herald selection. Low or zero gg/ee acceptance does not mean acquisition is incomplete. Incomplete points retain counts but supply no phase; the existing complete-axis fit and writeback rules still apply.

Preview every proposal before Apply Selected. Writeback uses the original Run or Load capture, even if you change aliases afterward. Writeback fails without that capture. Recalibrate after changing pulse or readout conditions. Herald selection does not establish gate fidelity.


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
