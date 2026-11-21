**Readers:** operator, agent. Sweep JPA pump source power and choose a measured setting using qubit readout separation.

# JPA pump power sweep

Set the JPA flux bias and pump RF frequency first. At each source power, the run takes two sets of shots: optional reset → no pulse → readout (g), or optional reset → qubit π pulse → readout (e). It changes power at the source, not the attenuation from the source to the chip, and does not toggle pump output.

## Set up the scan

Choose the qubit, resonator and JPA aliases at the current point, then choose **Defaults**. **JPA pump RF source** starts from `$jpa.rf_source`; select the connected source if the field is empty. **JPA pump source power** starts at `$jpa.pump_power` ± 5 dBm, with 101 points. Review the output state and narrow the bounds to powers safe for this device. **Scan-order seed** repeats the shuffled interior order for a nonnegative integer. Leave it blank for a new order; the endpoints stay first and last. Normal completion leaves the source at Stop.

Choose **Reps** for the shots per preparation and setting in each round, then **Rounds** for the number of such rounds. Defaults are 1000 × 10. The Analyze summary shows the Reps × Rounds from its source run. **Relaxation wait** starts at the qubit T1, or 30 µs if T1 is missing, and lets the system relax between shots. An invalid stored T1 leaves the wait blank; enter a calibrated delay.

Select **Qubit π pulse** from the qubit's X180 calibration and **Readout** from the resonator's readout calibration. Without a saved slot, Defaults uses the qubit RF frequency and drive channel for the π pulse, and the resonator frequency and readout channels for the resonator pulse. The readout trigger starts at the resonator flight time plus 0.015 µs. **Reset** is an optional pulse module before preparation. Leave it empty to rely on relaxation between shots.

## Choose a source power

The separation SNR is the distance between the g and e IQ means divided by their pooled width along the line joining the means. **IQ shape penalty** defaults to zero. Increasing it downweights unequal widths or skew along that axis. The resulting g/e separation score (a.u.) is neither a state population nor an assignment fidelity.

Live shows points as they arrive and counts sampled settings and retained shots. Analyze overlays the dashed line at the sampled power with the highest score. There are no pointwise shot-SEM bars or fitted errors. The analysis selects a measured peak and does not fit a model; an SEM of the mean IQ would not estimate uncertainty in the separation score. Rescan between the adjacent power samples around a repeatable interior peak. If the highest dot sits at a boundary, check whether the scan covers the peak before accepting it. For a flat or unstable trace, verify the pump and g/e preparation first. Subtract the previous source power from the selected power; a positive dB difference calls for an increase. Accepting the writeback updates `$jpa.pump_power` in dBm without changing the device.
