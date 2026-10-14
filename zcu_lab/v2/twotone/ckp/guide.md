## Behavior

CKP ground/excited-resolved spectroscopy: prepares the qubit in g and e, drives resonator and qubit probe tones concurrently over a two-dimensional frequency sweep, and fits the two resonator branches to extract dispersive shift chi, linewidth kappa, and the readout frequency. Runs on real hardware.

## Expects (parameter context)

Reads from the parameter context (all optional, seeding fresh cfg defaults): 'r_f' and 'rf_w' set the resonator sweep and pulse timing; 'q_f' sets the qubit-probe sweep; 'qub_ch' / 'qub_1_4_ch' / 'qub_4_5_ch' select the qubit drive channel; 'res_ch' / 'ro_ch' select readout channels; 'timeFly' seeds the readout trigger offset.

## Expects (ModuleLibrary)

Needs a calibrated qubit pi pulse (typically 'pi_amp' or 'pi_len'), an inline resonator-drive pulse, an inline qubit-probe pulse, and a pulse-readout module (typically 'readout_dpm' or 'readout_rf'). Optionally references a calibrated reset module.

## Typical writeback

Proposes fitted chi into parameter context 'chi', fitted kappa into 'rf_w', and fitted readout frequency into 'readout_f'. No ModuleLibrary writeback.

## Recommended

Run only after qubit frequency, pi pulse, resonator frequency, and resonator linewidth are calibrated. Review both g/e maps and fitted branches before applying writeback; a branch-selection or fit failure can produce plausible-looking scalar values.
