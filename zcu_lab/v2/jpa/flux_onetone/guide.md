**Readers:** operator, agent. Find resonator features that move with JPA bias, using the magnitude of the reflected IQ.

# JPA bias and readout spectrum

At each JPA flux-bias setting, sweep readout RF. Each shot is optional reset → readout at the selected RF. The run does not set pump RF, power or output; set those separately.

## Set the survey

- Bind the resonator and JPA. Confirm **JPA flux-bias source** (initially `$jpa.flux_dev`). Match **JPA bias unit** to that source. **JPA flux-bias value** starts at stored bias ± 0.005 in that unit, not Φ₀. Check approved bounds; the values go to the device unchanged.
- **Readout RF frequency** starts at the captured center ± 1.5 linewidths. Use absolute MHz. Check the wiring and ADC window of **Frequency-swept readout**. Its trigger starts at flight time + 0.015 µs and the sweep sets transmit and receive RF.

For tab setup, refreshing sources and saving, see [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md).

## Read the map and cuts

The map colors `|mean(I + iQ)|` (a.u.); blank cells are missing data. The bias axis displays recorded values as a.u., even if the form used A or V. The blue cut shows the latest sampled bias spectrum. Faint red cuts show up to four preceding bias positions. Use the cuts to locate the peak or notch. Analyze redraws the selected Run or Load; older results without readable settings still plot but have no configured Reps × Rounds line.

Extend a clipped RF edge; refine an interior feature with the neighboring bias and frequency values. There is no fit, shot SEM or writeback. Verify a candidate in a g/e separation sweep; the reflection magnitude alone does not measure calibrated gain, population or separation.
