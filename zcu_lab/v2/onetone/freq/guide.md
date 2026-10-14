## Behavior

One-tone resonator spectroscopy: sweeps the readout frequency and fits the resonator response to extract its frequency and linewidth. Runs on real hardware. Typically run after a coarse resonator search has placed you near the right frequency.

## Expects (parameter context)

Uses the tab's captured resonator parameters: freq and kappa seed the sweep; ext.default_probe_length and ext.time_of_fly seed readout timing; wiring.ch and wiring.ro_ch seed channels. Homophasal sampling requires explicit cfg calibration values for r_f, rf_w, and theta0, supplied directly or as expressions. Analysis optionally reads the route-matched resonator.ext.edelay prior from the same fixed capture.

## Expects (ModuleLibrary)

Starts with a fresh inline pulse readout and a const waveform. It does not adopt a calibrated library module or named waveform.

## Typical writeback

Proposes fitted values to resonator.freq, resonator.kappa, and resonator.ext.theta0 through the captured role binding. With a pulse-readout run snapshot, it also proposes resonator.readout.readout_rf from that template, changing only pulse/readout frequency while preserving gain, lengths, channels, and trigger timing. The compound resonator.ext.edelay proposal keeps delay and route together; it requires an identifiable absolute branch or a trusted prior/manual seed.

## Recommended

Analysis defaults to the hanger-model fit ('hm') with multiplicative amplitude-background fitting on. A sweep spanning a few linewidths around the known resonator frequency usually captures the dip cleanly; widen it if the resonator has drifted. Use homophasal sampling after a fit has written 'theta0' when you want equal resonator-circle phase spacing instead of a linear frequency grid. The homophasal calibration fields start empty; fill them with fitted values or expressions before selecting that mode. Electrical-delay mode 'auto' uses a route-matched calibration when available and otherwise runs bounded adaptive search; use 'calibrated' to require that prior or 'manual' to supply a seed.
