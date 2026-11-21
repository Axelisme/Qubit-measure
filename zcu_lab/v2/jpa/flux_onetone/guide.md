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

The map shows the magnitude of mean complex IQ as color (a.u.). It shows reflected signal, not state population, calibrated gain or g/e separation. The color scale follows measured data; blank columns show bias settings not yet sampled. Live reports how many bias spectra have data. The lower panel shows the latest bias spectrum to locate a dip or peak without estimating its position from color alone.

Trace a repeatable feature across the map. If it runs into a frequency boundary, widen the window toward the edge. To refine an interior feature, set the next bounds to the surrounding bias and frequency samples. There is no automatic bias selection. This spectral survey does not show pointwise shot-SEM bars. Confirm a candidate on a separate g/e separation scan before treating it as a readout optimum.
