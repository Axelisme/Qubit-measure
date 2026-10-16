## Behavior

Single-shot AC-Stark calibration: drives a cavity Stark tone and a qubit probe tone while sweeping cavity gain and qubit frequency (2D), classifying each shot in-program against the |g>/|e> IQ-cluster centres. Fits the Stark-shifted resonance versus gain² to extract the AC-Stark coefficient (photon number per gain²). Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay starts at 5.5 us. Qubit sweep is `$qubit.freq` - 700 to `$qubit.freq` + 100 MHz. Qubit probe length/mixer use `$qubit.module.x180.waveform.length` and `$qubit.module.x180.mixer_freq.` Pre/post delay are 1.0/`$resonator.kappa` and 0.5/`$resonator.kappa`; cavity tone length adds 1.0/`$resonator.kappa` to the pi length. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Reset and init start disabled. Cavity/qubit tones start from res_probe/pi_pulse with calibrated timing. Readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

Proposes the fitted AC-Stark coefficient into parameter context '`$qubit.ext.ac_stark_coeff`' (photon number per gain²); MIST reads it to rescale their x-axis to photon number.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'; run after the dispersive-shift experiment has set '`$qubit.ext.chi`' / '`$resonator.kappa`'. Sweep the Stark gain across the onset and the probe frequency around the qubit line.
