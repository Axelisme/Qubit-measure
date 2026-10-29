# Qutrit reference population

**Readers:** operator, agent. Acquire three labeled IQ references and inspect preparation-coordinate correction.

## Calibrate before Run

Open **New Tab → singleshot → qutrit_population**. Missing transition frequencies leave the Form Invalid.
The entry qubit frequency remains unset until you measure it. The Form does not calculate transition frequencies from the simulator.

Find 0–1 and 0–3 with ground-state twotone/freq spectroscopy.
Use the existing Rabi experiment at each measured carrier to determine Pi 01 and Pi 03 amplitude and duration.
Prepare level 3 with Pi 03 in twotone/freq's Init Pulse, then locate 3–2 and use prepared-state Rabi to calibrate Pi 32.
Enter those calibrations in the three pulse editors, including MHz frequency, gain, us duration and generator.
For integer-fluxonium, Pump uses generator 1, Stokes uses generator 2 and readout uses generator 0.
Pi 32 must target the configured charge line; it is not a fast-flux channel.

Use the environment's validated cavity/receiver settings: nominal carrier 7460 MHz, constant readout length 0.4 us, gain 0.1, ADC integration length 0.4 us and trigger offset 0 us.
These are synthetic input settings, not measured device calibration.
Set a relaxation interval that returns the system to the ground preparation before each repetition.
The simulator's acquire path starts each repetition from its configured thermal state; this does not establish hardware reset performance.

## Acquire and analyze

Run collects three rows in order: g has no preparation pulse, e applies Pi 01, f applies Pi 03 followed by Pi 32.
Each row contains **Shots per preparation** complex IQ values in mean-sample ADC units.
No reset or extra drive is inserted for g. The measurement plot shows the three acquired clouds.

Click Analysis to train a Euclidean nearest-centroid classifier on the first half of each row.
The remaining half evaluates assignment. An odd shot count gives the extra shot to evaluation.
Training confusion rows are requested preparation labels and columns are classified labels.
The Result fit figure shows held-out raw fractions, unclipped corrected fractions, training confusion and matrix quality.
The analysis also returns arrays, centroids, sample counts and diagnostics for the host's saved analysis summary.

Correction solves the transpose of the training confusion matrix.
Its coordinates are the reference preparation labels, not guaranteed pure energy states.
Negative and above-one estimates remain visible because finite calibration shots and preparation contamination can produce them.
Do not call the preparation-label assignment score pure readout fidelity.

## Quality and failure conditions

For the short-reference integer-fluxonium setting, inspect minimum pooled Mahalanobis separation >=2, macro preparation-label error <=3%, each label error <=6%, training confusion 2-norm condition <=1.5 and minimum singular value >=0.8.
These thresholds characterize this synthetic reference setting, not strong STIRAP/Raman drives or hardware.
An independent fixed-seed acquisition is needed to validate a selected readout configuration.

Analysis rejects nonfinite/partial IQ, fewer than six shots per row, invalid labels, indistinguishable centers, singular pooled IQ covariance or singular confusion.
It never substitutes simulator populations, a density matrix or replay truth.
Poor separation or high label error requires checking spectroscopy, pulse calibrations and receiver settings before interpreting correction.
A small matrix condition number alone does not establish state purity.
Reload experiments rebuilds this adapter; reopening the app retains entry data but does not invent missing pulse calibrations.
There is no MCP recipe for this experiment.
