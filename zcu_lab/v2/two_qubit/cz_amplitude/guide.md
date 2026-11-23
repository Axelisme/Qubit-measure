# CZ common-amplitude calibration

**Readers:** operator, agent. Find the common CZ gain multiplier for the first joint return.

## What the scan measures

Each shot runs **optional Reset → Xπ on both qubits → simultaneous CZ drives → Xπ on both qubits → terminal mux readout**. The two pulse timelines in each Xπ block start together and the block waits for both; individual pre-delays can shift the envelopes. Terminal P(g,g) is a return proxy, not gate fidelity, a direct leakage measurement, or evidence of a π conditional phase. Preparation, thermal population and readout errors contribute.

## Preparation and inputs

- Use calibrated **Qubit1 Xπ** and **Qubit2 Xπ** and a complete **Mux readout** with both tones' g/e circles and radii from GE Post under matching readout conditions. Match each qubit's position in the mux group's ordered list to its tone, not to an ADC number. See [position bindings and mux tone order](../guide.md#綁定位置與讀出-tones).
- **Two-line CZ drive** references the coupler's `cz_qubit1` and `cz_qubit2` slots. To try different settings for this run, choose **Convert to inline** on each pulse you need to edit. Without those slots, fill RF frequency, signed gain, Nyquist zone and envelope duration in the cosine templates; the generator comes from the wiring. Use independent generators and matching frequency, envelope length and pre-delay. Post-delays can differ; the sequence waits for both. Start from a pair chosen with [Chevron](../chevron/guide.md).
- For an Inline pulse, **Frequency** is the RF carrier and **Mixer freq** is the optional mixer frequency, both in MHz, not offsets. **Gen ch** identifies the drive generator; **NQZ** selects its DAC Nyquist zone. **Gain** is signed normalized amplitude and **Phase** is the carrier angle. **Waveform → Length** (µs) excludes the waits before and after the envelope. In both the Xπ blocks and the CZ pair, the next step waits for the later pulse end including its post-delay.
- Use **Common CZ gain multiplier → linrange** and fill **Start** and **Stop** from 0 through at least the first positive return. Choose **Expts**, including both endpoints, to resolve the oscillation. The `center` modes use a midpoint and full span instead; the `_step` modes use step size instead of Expts and may stop short of the requested upper endpoint. Check that the realized sweep still covers the return. Each multiplier scales both signed gains and leaves frequency, relative phase and waveform fixed.
- **Reps × Rounds** is the attempted shot budget per point. **Initial delay** is the wait before the first shot; set **Relaxation delay** after each shot using the source's qubit relaxation and cavity-emptying calibration.
- Blank **Reset** omits reset. For mux herald, use the same full readout and set `keep` by zero-based tone position, not ADC number; see [herald and post-delay checks](../guide.md#herald-reset-與獨立-heralded-入口).
- Recipe `two_qubit_cz_amplitude`: omitted `scale_range`, `scale_points`, `reps` and `rounds` keep the form values. `scale_range: null` clears Start and Stop; explicit null for `scale_points`, `reps` or `rounds` passes to the form. After ordered `overrides`, unrepaired required blanks fail `check_ready` before Run. An omitted or null `target` leaves the bindings unchanged.

## Read the figures

- **Run** connects measured terminal P(g,g) versus gain multiplier. Check both covered points and retained shots: full point coverage can precede the final round, and herald can reduce the retained count below the attempted count.
- **Analyze** shows measured points without a joining line when it draws the first-return fit. Without a fit, the line joins measured points. The dashed vertical line marks a selected return, not a fit. The joint-outcome panels connect measured probabilities on the same gain-multiplier axis as Run. Each panel fixes Qubit1 to g, e or other and shows three Qubit2 series. Use them to see which classes rise as P(g,g) falls. Five series contain `other`: one for each of the g and e panels and all three in the other panel. All nine count in the denominator. `Other` is circle rejection, not a measured higher-level population.
- In Run and both Analyze figures, measured probability points use ±1 shot SEM, `√[p(1−p)/(N−1)]` for `N ≥ 2` retained shots. In the joint-outcome panels a bar may be smaller than its marker at that probability scale. A near-zero point does not establish zero counts. Shot SEM is not single-shot spread, round-to-round variation or fit uncertainty. Period SE is a separate fit uncertainty.

## Fit the first complete return

The model is `offset + contrast × cos(2π × multiplier / period)`; `period` is the first positive return. Fit starts at 0 and uses the contiguous prefix of measured points with nonzero retained counts from completed rounds. A point does not need all planned rounds to enter the fit. A stopped suffix does not invalidate a full first cycle in that prefix. Missing coverage, zero contrast, a degenerate fit or nonfinite uncertainty blocks a proposal. A load without its source cfg cannot produce a fit; [legacy files](../guide.md#opendefaults-與人工-apply) may need the version that wrote them.

Compare the curve with the sampled points. R² measures explained variation, normalized RMS is residual RMS divided by the observed probability span, and relative fit SE is each parameter's SE divided by its absolute fitted value. There is no separate numeric quality cutoff. Low nonzero contrast can still produce a candidate. Compare the fitted peak-to-trough change, `2 × |contrast|`, with the shot SEM and check that the points follow a full oscillation. If noise hides the return, collect more shots or sample the first cycle more densely before applying a multiplier. If the points show a systematic shape the fit misses, check the model and drive settings rather than relying on a small period SE.

## Preview and selected apply

Writeback proposes `new signed gain = saved Run signed gain × fitted multiplier` for both `cz_qubit1` and `cz_qubit2` pulses and offers their coupler slots as independent rows. The source is the pulse used in that Run, including Inline edits, not the current form or the original library slot. The proposal changes no other pulse field. Without the Run or Load capture there is no writeback. After accepting a supported return, check [conditional phase](../conditional_phase/guide.md); recheck amplitude after a frequency change and remeasure local Z after changing the gate or bias.

For common GUI actions, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).
