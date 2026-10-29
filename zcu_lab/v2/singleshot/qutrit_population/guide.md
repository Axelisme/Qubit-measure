# Qutrit reference population

**Readers:** operator, agent. Acquire three labeled IQ references and inspect preparation-coordinate correction.

## Calibrate before Run

Open **New Tab → singleshot → qutrit_population**. Missing transition frequencies leave the Form Invalid.
The entry qubit frequency remains unset until you measure it. The Form does not calculate transition frequencies from the simulator.

Find 0–1 and 0–3 with ground-state twotone/freq spectroscopy.
Use the existing Rabi experiment at each measured carrier to determine Pi 01 and Pi 03 amplitude and duration.
Prepare level 3 with Pi 03 in twotone/freq's Init Pulse, then locate 3–2.
Open New Tab → twotone → rabi → amp_rabi and set Init Pulse to the calibrated Pi 03 on generator 1.
Set Rabi Pulse to generator 2, the measured 3–2 carrier and a fixed duration. Sweep Gain, inspect the oscillation, then click Analysis → Analyze to determine the Pi 32 gain.
See the [amplitude Rabi guide](../../twotone/rabi/amp_rabi/guide.md#prepared-state-amplitude-rabi) for the pulse fields and None ground-state path.
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

## Apply calibration to readout

Before Run, save the tested readout as a library module and bind the resonator's readout slot to its key.
Select that captured readout in the Form without changing its non-classifier settings.
After **Analysis → Analyze**, inspect the module proposal's g/e/f centers and training confusion, then click **Apply Selected**.
Apply replaces the complete captured readout module but changes only its classifier. It preserves the waveform key and all other fields.
It does not create a missing library target or choose one from the current Form.
Analyze reports a failure if the Run had no captured target, the slot is unresolved, or any non-classifier setting differs from the captured library module.
Save and bind the intended settings, then Run again to fix those failures.

Reopen a consumer after Apply to capture the new library calibration.
The classifier uses Euclidean nearest centers with no rejection region. Exact ties choose g, then e, then f.
Classified acquire stores two columns, g and e. Recover f as `1-g-e` for this classifier only.
Circle calibration's third class remains other. Circle-only consumers and heralding reject the three-center classifier.
Saved run cfg retains its own centers and training confusion. Later live-library edits do not change that run's classification or correction.
There is no automatic calibration invalidation. Repeat reference acquisition and Apply after changing readout pulse, ADC window, receiver or preparation settings.
Training confusion includes preparation contamination, so correction reports preparation-label coordinates, not pure-state fidelity.

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
