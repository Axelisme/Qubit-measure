# Heralded two-qubit paired RB and CZ-interleaved RB

**Readers:** operator, agent. Measure gg- and ee-conditioned returns and test whether the two groups support a shared decay per arm.

Select `two_qubit/rb_heralded` and use the pulses, ordered local-Z pair and depth/seed settings in [Before run](guide.md#before-run), which also covers recipe inputs. Blank **Reset** omits optional preparation; this entry always supplies its own herald and rejects a herald reset module.

## Expanded pulse and readout fields

Keep the calibrated waveform, routing and mixer settings. **Frequency** and **Mixer freq** are absolute MHz; the mixer field may be blank on an unmixed drive. **Gen ch** is the DAC channel, **NQZ** its Nyquist zone and **Ro ch** the ADC channel. In **Mux readout**, **Length** is the envelope length, **RO length** the integration duration, and **Pre-delay** and **Trig offset** set the DAC and ADC starts from the readout step's origin. Times are in µs. A tone label is not its binding. Circle centers and radius share normalized ADC units; RB does not use the optional confusion matrix to correct counts.

## Shot sequence

**Optional ordinary Reset → joint herald readout → random Clifford prefix (one added CZ per interleaved Clifford) → inverse recovery → joint terminal readout**. Both reads share full mux settings and calibrated GE circles and radii. Use [the same paired prefixes and local frames as ordinary RB](guide.md#sequences-and-cost). Compare terminal gg within gg herald and terminal ee within ee herald. This measures classified return, not state purity or fidelity.

## Wait after herald

Set **Mux readout** post-delay to start at `5/(2π·κ)` µs using the smallest FWHM linewidth κ of the shared readout, in MHz. The wait starts at the end of the waveform and applies to both reads. Check it in [Joint readout](../joint_readout/guide.md) with gg herald: sum the target terminal-e columns in the row with target Xπ and spectator I. Repeat with only a longer post-delay. If the e fraction still increases, adjust the wait before repeating RB. Residual photons can shift the qubit frequency; the classified e fraction is not gate fidelity.

## Joint decay and uncertainty

- **Conditional return** (Run and first Analyze) pools by arm and depth over available seeds. The panels show P(gg terminal | gg herald) and P(ee terminal | ee herald); each group's denominator includes all its terminal classes. The live shot count spans all herald outcomes, not just the two fit groups.
- **gg decay fit** and **ee decay fit** show equal-weight seed means and fitted curves. Each group needs at least four distinct depths and two seeds with data at every depth for both arms; groups may use different seeds. A single `p` per arm is shared across groups, but the amplitude and offset of `A p^depth + B` are separate.
- Capped bars show shot SEM: 0/1 sample SD divided by √N for the indicated herald group. Pale bars and the fit point bars show across-seed SD (complete seeds for the fit); neither is a round spread or fit uncertainty. The 1/e depth ± is fit SE in Cliffords, shared across heralds for each arm. Relative SE, R² and normalized RMS mean the same as in [ordinary RB](guide.md#interpreting-the-result); the groups have separate residual metrics.
- CZ error is `3/4 × (1 − p_interleaved/p_reference)`. The 95% CI resamples whole paired seed identities from the union of valid seeds, keeping both arms and available groups together. At least 90% of draws must give valid fits. It does not cover readout or gate-dependent systematic error. Check both herald fits and the depth-window warnings; if the decay is unresolved, adjust depths as in ordinary RB, not CZ gain or frequency. Preserve a signed negative estimate as a diagnostic. There is no writeback.

## Herald fractions

**Herald outcome fractions** shows nine probabilities and shot SEM against depth for each arm. Each panel fixes Qubit1 and uses series for Qubit2. Every probability divides by all herald shots at that arm/depth. Check whether gg or ee has too few shots, or other dominates; low gg fraction is acceptance, not purity. Check GE calibration before spending more shots on a low-contrast group.

See the [legacy load note](../guide.md#opendefaults-與人工-apply) and [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
