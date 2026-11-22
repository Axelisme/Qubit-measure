**Readers:** operator, agent. Compare the resonator's reflected IQ spectrum with the JPA pump off and on.

# JPA pump check

Pump off → optional reset and swept readout; then pump on → repeat. Pump RF, power and bias stay at their existing settings. A normal Run **leaves the pump on**.

## Set up the comparison

- Bind the resonator and JPA. Check **JPA pump RF source** (initially `$jpa.rf_source`) and the active pump settings before toggling output.
- **Readout RF frequency** starts at the captured center ± 1.5 linewidths. Enter absolute MHz and include both shoulders. The sweep sets transmit and receive RF.
- Verify the calibrated wiring and ADC window in **Frequency-swept readout**. The trigger starts at flight time + 0.015 µs. **Reset** may be omitted.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the plots

The two Live and Analyze traces show `|mean(I + iQ)|` (a.u.) against absolute RF. Blank frequencies are unmeasured. Compare the feature's center, width and contrast in the two pump states. The traces do not measure calibrated gain in dB or g/e separation. There is no fit, shot SEM or writeback.

Narrow the window to refine a feature with both shoulders in view. To choose a pump setting for qubit readout, run a separate g/e separation sweep.
