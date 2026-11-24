# Heralded two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Read the pooled return for gg, ge, eg and ee and check the shared decay in each group.

Select `two_qubit/rb_heralded` and use the pulses, ordered local-Z pair and depth/seed settings in [Before run](guide.md#before-run), which also covers recipe inputs. Blank **Reset** omits optional preparation; this entry always supplies its own herald and rejects a herald reset module.

## Expanded pulse and readout fields

Keep the calibrated waveform, routing and mixer settings. **Frequency** and **Mixer freq** are absolute MHz; the mixer field may be blank on an unmixed drive. **Gen ch** is the DAC channel, **NQZ** its Nyquist zone and **Ro ch** the ADC channel. In **Mux readout**, **Length** is the envelope length, **RO length** the integration duration, and **Pre-delay** and **Trig offset** set the DAC and ADC starts from the readout step's origin. Times are in µs. A tone label is not its binding. Circle centers and radius share normalized ADC units; RB does not use the optional confusion matrix to correct counts.

## Shot sequence

**Optional ordinary Reset → joint herald readout → random Clifford prefix (one added CZ per interleaved Clifford) → inverse recovery → joint terminal readout**. Both reads share full mux settings and calibrated GE circles and radii. Use [the same paired prefixes and local frames as ordinary RB](guide.md#sequences-and-cost). A return means the terminal class matches the herald class: gg→gg, ge→ge, eg→eg or ee→ee. A shot with either qubit heralded Other does not enter the return analysis. This measures classified return, not state purity or fidelity.

## Wait after herald

Set **Mux readout** post-delay to start at `5/(2π·κ)` µs using the smallest FWHM linewidth κ of the shared readout, in MHz. The wait starts at the end of the waveform and applies to both reads. Check it in [Joint readout](../joint_readout/guide.md) with gg herald: sum the target terminal-e columns in the row with target Xπ and spectator I. Repeat with only a longer post-delay. If the e fraction still increases, adjust the wait before repeating RB. Residual photons can shift the qubit frequency; the classified e fraction is not gate fidelity.

## Joint decay and uncertainty

- **Conditional return** (Live and first Analyze) shows one measured series for Reference and one for CZ interleaved. At each depth, divide the matching returns by all shots heralded gg, ge, eg or ee across available seeds. All terminal classes, including Other, count in that denominator. Without any shots in those four groups, the point and bars are absent.
- **Conditional return** in Analyze overlays one fit per arm and reports the joint decay `p`, CZ error and its seed CI. At each measured depth, the line mixes the four group fits with weights equal to the observed herald shot fractions for that arm. Between measured depths with eligible shots, it interpolates the proportions linearly. With constant proportions, this gives a single exponential with the joint fit's `p`. With changing proportions, the combined line need not be a single exponential. This overlay does not refit the pooled points. The caption lists normalized residual RMS for each herald and arm from the joint fit, not from a fit to pooled points.
- **Conditional return by herald** shows four panels. Each arm pools the available shots for that herald, including shots from incomplete seeds. Capped bars show the group's shot SEM. Pale bars show its across-seed SD. Compare groups at matching depths to see which ones account for changes in the pooled return.
- The **decay fit diagnostic** for each herald shows equal-weight seed means and the group fit. Fitting requires at least four distinct depths and at least two seeds per group with both arms complete at every depth. The groups need not use the same seeds. Each arm shares one `p` across the four groups. Each group has its own amplitude and offset in each arm.
- Capped bars in **Conditional return** show shot SEM: the 0/1 return sample SD divided by √N for the combined count. Pale bars show across-seed SD of the pooled return within each seed. Point bars in the decay fit diagnostics show across-seed SD within each herald group, using complete seeds. None of these bars shows round spread or fit uncertainty. In the diagnostics, the error on 1/e depth is the fit SE in Cliffords, shared across herald groups for each arm. Relative SE, R² and normalized RMS mean the same as in [ordinary RB](guide.md#interpreting-the-result). The residual metrics are separate for each group.
- CZ error is `3/4 × (1 − p_interleaved/p_reference)`. For the 95% CI, Analyze draws whole seed identities from the union of the four complete cohorts. Both arms, all depths and every available group travel with the same seed identity. Repeated draws keep their weight. Each draw must support the same four-group fit, and at least 90% must succeed. The interval does not cover readout or gate-dependent systematic error.
- A group with too few complete seeds, a failed joint fit or an unstable seed CI leaves the observed returns and their available bars visible. Analyze gives the reason and shows no `p`, CZ error, fit lines or fit quality. Check all four group diagnostics and depth-window warnings. If the decay is unresolved, adjust depths as in ordinary RB. Preserve a signed negative estimate as a diagnostic. There is no writeback.

## Herald fractions

**Herald outcome fractions** shows nine probabilities and shot SEM against depth for each arm. Each panel fixes Qubit1 and uses series for Qubit2. Every probability divides by all herald shots at that arm/depth. Check which groups have few shots and whether Other dominates. A low fraction gives fewer shots for that group; it does not measure state purity. Check GE calibration before spending more shots on a low-contrast group.

See the [legacy load note](../guide.md#opendefaults-與人工-apply) and [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
