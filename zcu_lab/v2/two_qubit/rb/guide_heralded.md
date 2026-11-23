# Heralded two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Read the pooled gg/ee return and check the shared decay in each herald group.

Select `two_qubit/rb_heralded` and use the pulses, ordered local-Z pair and depth/seed settings in [Before run](guide.md#before-run), which also covers recipe inputs. Blank **Reset** omits optional preparation; this entry always supplies its own herald and rejects a herald reset module.

## Expanded pulse and readout fields

Keep the calibrated waveform, routing and mixer settings. **Frequency** and **Mixer freq** are absolute MHz; the mixer field may be blank on an unmixed drive. **Gen ch** is the DAC channel, **NQZ** its Nyquist zone and **Ro ch** the ADC channel. In **Mux readout**, **Length** is the envelope length, **RO length** the integration duration, and **Pre-delay** and **Trig offset** set the DAC and ADC starts from the readout step's origin. Times are in µs. A tone label is not its binding. Circle centers and radius share normalized ADC units; RB does not use the optional confusion matrix to correct counts.

## Shot sequence

**Optional ordinary Reset → joint herald readout → random Clifford prefix (one added CZ per interleaved Clifford) → inverse recovery → joint terminal readout**. Both reads share full mux settings and calibrated GE circles and radii. Use [the same paired prefixes and local frames as ordinary RB](guide.md#sequences-and-cost). A return means terminal gg after gg herald or terminal ee after ee herald. This measures classified return, not state purity or fidelity.

## Wait after herald

Set **Mux readout** post-delay to start at `5/(2π·κ)` µs using the smallest FWHM linewidth κ of the shared readout, in MHz. The wait starts at the end of the waveform and applies to both reads. Check it in [Joint readout](../joint_readout/guide.md) with gg herald: sum the target terminal-e columns in the row with target Xπ and spectator I. Repeat with only a longer post-delay. If the e fraction still increases, adjust the wait before repeating RB. Residual photons can shift the qubit frequency; the classified e fraction is not gate fidelity.

## Joint decay and uncertainty

- **Conditional return** (Live and first Analyze) shows one measured series for Reference and one for CZ interleaved. At each depth, divide the total gg→gg and ee→ee returns by all shots heralded gg or ee across available seeds. All terminal classes count in that denominator. If neither herald group has shots at a depth, the point and bars are absent. The Live progress count still includes all herald outcomes.
- **Conditional return** in Analyze overlays one fit per arm and reports the joint decay `p`, CZ error and its seed CI. At each measured depth, the line mixes the two group fits in the same gg/ee shot proportions as the data point. Between measured depths with eligible shots, it interpolates the proportions linearly. With constant proportions, this gives a single exponential with the joint fit's `p`. With changing proportions, the combined line need not be a single exponential. This overlay does not refit the pooled points.
- **Conditional return by herald** shows separate gg and ee panels in Analyze. Each arm's points pool shots in that herald group, including available shots from incomplete seeds. Capped bars show that group's shot SEM. Pale bars show that group's across-seed SD.
- **gg decay fit diagnostic** and **ee decay fit diagnostic** show equal-weight seed means and group fits. Each group needs at least four distinct depths and two seeds with data at every depth for both arms. The two groups may use different seeds. The joint fit uses one `p` per arm and separate amplitudes and offsets for each group.
- Capped bars in **Conditional return** show shot SEM: the 0/1 return sample SD divided by √N for the combined count. Pale bars show across-seed SD of the pooled return within each seed. Point bars in the decay fit diagnostics show across-seed SD within each herald group, using complete seeds. None of these bars shows round spread or fit uncertainty. In the diagnostics, the error on 1/e depth is the fit SE in Cliffords, shared across herald groups for each arm. Relative SE, R² and normalized RMS mean the same as in [ordinary RB](guide.md#interpreting-the-result). The residual metrics are separate for each group.
- CZ error is `3/4 × (1 − p_interleaved/p_reference)`. The 95% CI resamples whole paired seed identities from the union of valid seeds, keeping both arms and available groups together. At least 90% of draws must give valid fits. It does not cover readout or gate-dependent systematic error. Check both herald fits and the depth-window warnings; if the decay is unresolved, adjust depths as in ordinary RB, not CZ gain or frequency. Preserve a signed negative estimate as a diagnostic. There is no writeback.

## Herald fractions

**Herald outcome fractions** shows nine probabilities and shot SEM against depth for each arm. Each panel fixes Qubit1 and uses series for Qubit2. Every probability divides by all herald shots at that arm/depth. Check whether gg or ee has too few shots, or other dominates; low gg fraction is acceptance, not purity. Check GE calibration before spending more shots on a low-contrast group.

See the [legacy load note](../guide.md#opendefaults-與人工-apply) and [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
