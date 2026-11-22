# Two-tone readout-chain sweep

**Readers:** operator, agent. Locate qubit spectral lines that alter the statistics of repeated resonator reads, even when mean IQ changes little.

## Sequence and key settings

One dwell at one drive frequency is **Reset** (optional) → **Init Pulse** (optional, once per dwell) → 48 cycles of **Probe Pulse** then readout without resetting between cycles. The next probe follows the prior readout pulse, its post-delay and the probe pre-delay. Check that integration windows on the same ADC do not overlap. The cycle count is fixed.

- Calibrate readout first and select a pulse readout, not mux. **Probe RF frequency** starts at `$qubit.freq ± 20 × $qubit.kappa` in MHz; enter the window directly for a first search.
- A fast sweep visits every RF point once. **Reps** sets fast sweeps per round and must be at least 12, so each RF point has enough dwells for the first live estimate. **Rounds** sets averaging passes. Smaller Reps updates Live more often. **Stop** keeps completed rounds; stopping before the first completes fails the Run.
- Set **Expected center** to select the nearest candidate, or leave it blank for the most prominent one. Set candidate gates with **Minimum prominence / signal range**, **Minimum height above median / noise sigma**, **Minimum half-prominence width**, and **Minimum peak separation**. Keep **Model type** at `lor`: a nonnegative sigma curve cannot show signed sinc sidelobes.

See [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Live shows RF frequency against evidence above the median background (σ), estimated from IQ means, second moments and inter-read correlations. The dashed **5% false-mark level** means that in a noise-only scan the chance any point crosses it is about 5%. This is an evidence statistic, not population; there is no pointwise shot SEM.

Analyze refits the same curve. Gray triangles are other candidates; the dotted line marks **Peak center**. The fit window ends halfway to neighbors and needs at least six samples. If it spans at most half the scan, the overview shades it and a second panel zooms in. Candidate SNR is peak height above the median in noise σ, not fit quality. No candidate passing the gates fails Analyze, not Run.

## Next step

Read **Peak center** and **Sigma-peak FWHM** in MHz with their fit errors. Noise-floor clipping can make a weak peak appear narrower. To calibrate `$qubit.freq` and `$qubit.kappa`, run `twotone/freq` around the center; this experiment offers no writeback.

For `twotone_chain`, null disables **Reset** but null for **Readout** or **Probe Pulse** fails during selection, before overrides. Null Gain, Length, Reps or Rounds leaves a blank; unresolved required blanks fail readiness before Run. Null **Expected center** selects the most prominent peak at Analyze.
