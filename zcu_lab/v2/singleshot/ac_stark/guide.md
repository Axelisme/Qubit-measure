## Behavior

Single-shot AC-Stark calibration: drives a cavity Stark tone and a qubit probe tone while sweeping cavity gain and qubit frequency (2D), classifying each shot in-program against the |g>/|e> IQ-cluster centres. Fits the Stark-shifted resonance versus gain² to extract the AC-Stark coefficient (photon number per gain²). Runs on real hardware.

## Expects (parameter context)

Run classification defaults reference `$qubit.ext.readout_cal.g_center`, `$qubit.ext.readout_cal.e_center` and `$qubit.ext.readout_cal.radius`. Probe frequency/channel use `$qubit.freq` and `$qubit.wiring.L01.ch`; probe length/mixer use `$qubit.module.x180.waveform.length` and `$qubit.module.x180.mixer_freq`. Cavity timing uses `$resonator.kappa`. Missing aliases or calibration leave cfg Invalid. Analyze calibration sources are not yet migrated to the historical Run contract.

## Expects (ModuleLibrary)

Uses `$qubit.module.readout` for SMART readout. Cavity and qubit probe tones are inline with canonical scalar calibration references; reset and init pulse start disabled.

## Typical writeback

Proposes the fitted AC-Stark coefficient into parameter context 'ac_stark_coeff' (photon number per gain²); the MIST experiments read it to rescale their x-axis to photon number.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'; run after the dispersive-shift experiment has set 'chi' / 'rf_w'. Sweep the Stark gain across the onset and the probe frequency around the qubit line.
