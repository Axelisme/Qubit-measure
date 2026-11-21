**Readers:** operator, agent. Sweep the JPA pump RF frequency and choose a measured setting using qubit readout separation.

# JPA pump frequency sweep

At each pump frequency, the run measures two preparations: optional reset → no pulse → resonator readout (g), or optional reset → qubit π pulse → resonator readout (e). The run uses the flux bias and pump source power already set on the devices. It tunes only the pump RF frequency and does not toggle pump output.

## Set up the scan

Open or choose **Defaults** on the selected point. The qubit and resonator aliases select the preparation and readout calibrations. The `jpa` alias selects the pump settings. Check the following before Run:

- **JPA pump RF source** starts from `$jpa.rf_source`. Select the connected pump source and confirm its output, power and bias are safe for the scan. If left empty, the run acquires at the same physical pump setting for every nominal coordinate. It cannot calibrate pump frequency in that case.
- **JPA pump RF frequency** is the absolute pump output frequency (MHz), not the readout RF or a mixer IF. **Start** and **Stop** start at 0.98 and 1.02 times `$jpa.pump_freq`; **Expts** starts at 101 samples. Check the endpoints against the desired range and the hardware settings.
- **Scan-order seed** accepts a nonnegative integer to repeat the interior scan order. Leave it blank to draw a new order. The endpoints are always measured first and last. Normal completion with tuning leaves the RF source at Stop.

Choose **Reps** for the shots per preparation and setting in each round, then **Rounds** for the number of such rounds. Defaults are 1000 × 10. The Analyze summary shows the Reps × Rounds from its source run. **Relaxation wait** starts at the qubit T1, or 30 µs if T1 is missing, and lets the system relax between shots. An invalid stored T1 leaves the wait blank; enter a calibrated delay.

Select **Qubit π pulse** from the qubit's X180 calibration and **Readout** from the resonator's readout calibration. Without a saved slot, Defaults uses the qubit RF frequency and drive channel for the π pulse, and the resonator frequency and readout channels for the resonator pulse. The readout trigger starts at the resonator flight time plus 0.015 µs. **Reset** is an optional pulse module before preparation. Leave it empty to rely on relaxation between shots.

If you edit an inline π pulse or readout, RF frequency is the actual carrier and mixer frequency, when set, configures the mixing stage. Gain is a fraction of full-scale output; phase sets the pulse phase in degrees. Select the calibrated generator, Nyquist zone and waveform for the path. Readout channel selects the ADC. ADC length sets its IQ integration window; trigger offset sets when that window starts relative to the readout. Readout pre-delay precedes the pulse and post-delay gives the resonator time to ring down. Start with a post-delay of 5/(2π·κ), with κ in MHz and the wait in µs. A saved classifier does not enter this moment-based analysis.

## Read the plots

The separation SNR is the distance between the g and e IQ means divided by their pooled width along the line joining the means. **IQ shape penalty** defaults to zero. Increasing it downweights unequal widths or skew along that axis. The resulting g/e separation score (a.u.) is neither a state population nor an assignment fidelity.

Live shows one dot per sampled pump RF frequency and reports how many settings have a score and how many shots from both preparations have contributed so far. The dots are not connected because the interior frequencies are shuffled. Analyze shows the same dots and a dashed line at the highest sampled score. The legend reports that pump RF frequency, not the resonator frequency. There are no pointwise shot-SEM bars or fitted errors. The analysis selects a measured peak and does not fit a model; an SEM of the mean IQ would not estimate uncertainty in the separation score.

Use the dashed-line frequency as a candidate only if the scan used pump tuning and the peak repeats. For an interior peak, rescan around it using the adjacent sampled frequencies as the new bounds. For a rising edge, move the window in that direction only within safe pump settings. At zero IQ shape penalty, a score below 1 means the IQ means are less than one pooled width apart. A weak or noisy separation trace does not determine a pump optimum. Check the g/e preparations and independently verify pump output before rescanning. To compare with the stored calibration, subtract the old pump frequency from the candidate in MHz; a positive difference means increase it. Accepting the writeback updates `$jpa.pump_freq`, not the live device.
