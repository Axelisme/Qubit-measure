**Readers:** operator, agent. Find bias-dependent resonator features in a JPA bias and readout-frequency survey.

# JPA bias and readout spectrum

The run steps the JPA flux bias. At each bias it acquires a resonator frequency sweep with the sequence optional reset → readout. It does not set the pump frequency, power or output. Set these separately and verify the safe bias range before Run.

## Set up the survey

Choose the resonator and JPA aliases at the current point, then choose **Defaults**.

- **JPA flux-bias source** starts from `$jpa.flux_dev`. Select the connected DC source. **JPA bias unit** starts at A; match it to the source. Values are sent unchanged. **JPA flux-bias value** starts at `$jpa.flux` ± 0.005 in the chosen unit with 101 samples. This scans a source setting, not flux in Φ₀.
- **Readout RF frequency** starts at `$resonator.freq` ± 1.5 × `$resonator.kappa` with 101 samples. This is the absolute resonator probe RF (MHz).
- **Frequency-swept readout** starts from the resonator wiring and timing, with the trigger at the flight time plus 0.015 µs. The sweep sets transmit and receive RF frequency. Choose a calibrated readout for this route. Select **Reset** from the qubit's optional reset module, or leave it empty.
- **Reps** sets shots per frequency per round (1000 by default); **Rounds** sets rounds at each bias (10). **Relaxation wait** is the delay after each readout (0.1 µs by default).

## Read the plots

After **Run** or a successful **Load**, choose **Analyze** to redraw that result. Load alone does not open a plot. The summary lists the source's **Reps × Rounds** if its saved settings can be read. Older results without readable settings still plot.

The map colors the magnitude of mean complex IQ (a.u.). It shows reflected signal, not state population, calibrated gain or g/e separation. The bias axis shows the recorded device values in native coordinates (a.u.); the form's A/V/native choice controls the source and does not rescale or relabel the plot. The color scale follows the measured data. Blank cells are missing samples, not zeros. The live title counts bias spectra with data.

The lower panel shows the latest sampled bias spectrum in blue. The four preceding bias positions appear in faint red where data exist. Use these cuts to locate a dip or peak without estimating its position from color alone.

Trace a repeatable feature across the map. If it runs into a frequency boundary, widen the window toward the edge. To refine an interior feature, set the next bounds to the surrounding bias and frequency samples. Analyze does not fit or select a bias, and it does not write back settings. You decide which bias to check next. This spectral survey does not show pointwise shot-SEM bars. Confirm a candidate on a separate g/e separation scan before treating it as a readout optimum.
