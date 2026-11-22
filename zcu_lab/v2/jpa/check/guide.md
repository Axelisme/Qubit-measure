**Readers:** operator, agent. Compare the resonator reflection spectra with the JPA pump off and on.

# JPA pump check

The run turns the pump off, then takes an optional reset → frequency-swept readout spectrum. It turns the pump on and repeats the same sequence. Pump frequency, power and flux bias stay at their existing settings. **This leaves the pump on after a normal run.** Verify its settings and the selected device before Run.

## Set up the comparison

Choose the resonator and JPA aliases at the selected point, then choose **Defaults**.

- **JPA pump RF source** starts from `$jpa.rf_source`. Select the connected source whose output to toggle.
- **Readout RF frequency** uses absolute RF in MHz. The initial bounds are `$resonator.freq` ± 1.5 × `$resonator.kappa` with 101 samples. Use a window wide enough to see the feature and both shoulders.
- **Frequency-swept readout** starts from the resonator's wiring and timing; its trigger starts at the flight time plus 0.015 µs. The sweep sets transmit and receive RF frequency. Choose a calibrated readout for this route. Select **Reset** from the qubit's optional reset module or leave it empty.
- **Reps** and **Rounds** start at 1000 shots per frequency per round and 10 rounds per pump setting. **Relaxation wait** starts at 0.5 µs and sets the delay after readout. The Analyze summary shows Reps × Rounds for the plotted run.

## Read the plots

The two labeled traces in Live and Analyze use the same RF axis and the magnitude of mean complex IQ, in a.u. This is a reflected readout signal, not a state population or calibrated gain in dB. Blank frequencies have no data yet; Live counts the sampled frequency and pump-state pairs. Compare the feature's position, width and contrast between the pump conditions. A change is evidence that the pump affects the reflection, but does not by itself identify the pump setting with the best g/e separation. Rescan with narrower frequency bounds around the feature if more resolution is needed. Use a g/e separation sweep to choose a pump setting. The spectra have no fit, shot-SEM bars or automatic writeback.
