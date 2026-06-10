# Magic names — measurement quantity glossary

The naming conventions for the `md` (MetaDict) values that flow through the single-qubit
measurement workflow: what each name means, its unit, which experiment produces it, and how
the names relate to one another.

**Boundary.** Per-experiment defaults and recommended sweep ranges are NOT here — they come
from `gui_adapter_guide` (each adapter declares `expects_md` and `recommended`, and computes
its own default cfg from `md`). This file owns only what no single experiment owns: the
global vocabulary and the cross-experiment relations / physics constants in the last section.

Channel names (`res_ch`, `ro_ch`, `qub_4_5_ch`, `lo_flux_ch`, …) are rig wiring and live in
`measure_setup.yaml` (see the `measure-setup` skill), not here.

Units: frequencies MHz, times µs, gains normalized 0–1, flux in the device unit (A in current
mode), JPA power dBm, IQ-plane quantities arb — unless noted.

## Resonator / readout

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `r_f` | resonator (bare) frequency | MHz | onetone `FreqExp` | reset_f = r_f − q_f; JPA pump ≈ 2·r_f |
| `rf_w` | resonator linewidth κ | MHz | onetone `FreqExp` / CKP / dispersive | ring-down τ = 1/(2π·rf_w); post_delay = 5τ |
| `chi` | dispersive shift χ | MHz | `DispersiveExp` / CKP | readout contrast |
| `readout_f` | readout center frequency | MHz | CKP | readout / probe pulse freq |
| `res_probe_len` | resonator probe pulse length | µs | set by hand | onetone ro_length = res_probe_len − 0.1 |
| `timeFly` | readout time-of-flight delay | µs | `LookbackExp` | ro trig_offset ≈ timeFly |
| `best_ro_freq` / `best_ro_gain` / `best_ro_length` | optimized readout freq / gain / ADC length | MHz / — / µs | `ro_optimize.*` | the `readout_dpm` module |

## Qubit

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `q_f` | qubit 0→1 frequency | MHz | twotone `FreqExp` / T2Ramsey / predictor | reset_f = r_f − q_f |
| `qf_w` | qubit linewidth | MHz | twotone `FreqExp` | — |

## Flux

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `flx_half` | half-flux (Φ₀/2) sweet spot | A | onetone/twotone `FluxDep` | predictor.flux_half |
| `flx_int` | integer-flux sweet spot | A | onetone/twotone `FluxDep` | predictor / bias |
| `flx_period` | flux period | A | derived | = 2·abs(flx_int − flx_half) |
| `flx_bias` | predictor flux-bias correction | A | `predictor.calculate_bias` | predictor.update_bias |

## Rabi / pulse

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `pi_len` / `pi2_len` | π / π-half pulse length | µs | `LenRabiExp` | `pi_len` / `pi2_len` modules |
| `pi_gain` / `pi2_gain` | π / π-half pulse gain | — | `AmpRabiExp` / ZigZag | `pi_amp` / `pi2_amp` modules |
| `rabi_f` | Rabi frequency | MHz | `LenRabiExp` | bath-reset qubit-tone gain |

## Reset

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `reset_f` | single-tone (sideband) reset freq | MHz | reset `single_tone.FreqExp` | seed: r_f − q_f |
| `reset_f1` | dual-tone reset freq 1 | MHz | reset `dual_tone` | seed: predict_freq(transition) |
| `reset_f2` | dual-tone reset freq 2 | MHz | reset `dual_tone` | seed: abs(r_f + predict_freq(transition)) |
| `resetf1_w` | reset-f1 linewidth | MHz | dual reset freq1 | — |
| `bathreset_freq` / `bathreset_gain` | bath-reset cavity-tone freq / gain | MHz / — | reset `bath.FreqGainExp` | `reset_bath` module |

## JPA

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `best_jpa_freq` | JPA pump frequency | MHz | `jpa.FreqExp` / AutoOptimize | ≈ 2·r_f + offset |
| `best_jpa_flux` | JPA flux bias | A | `jpa.FluxExp` / AutoOptimize | — |
| `best_jpa_power` | JPA pump power | dBm | `jpa.PowerExp` / AutoOptimize | — |
| `cur_jpa_A` | current JPA flux setting | A | device read | — |

## Coherence / time-domain

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `t1` (`t1err`) | T1 relaxation | µs | `T1Exp` | relax_delay ≈ 5·t1 |
| `t2r` (`t2r_err`) | T2 Ramsey | µs | `T2RamseyExp` | — |
| `t2e` (`t2e_err`) | T2 echo | µs | `T2EchoExp` | — |
| `t1_with_tone` | T1 under a readout tone | µs | `T1WithToneExp` | MIST / readout-length sizing |
| `ac_stark_coeff` | resonator photon-number coefficient | n̄/gain² | `AcStarkExp` | n̄ = ac_stark_coeff·gain²; MIST / T1-with-tone |

## Single-shot / discrimination

| name | meaning | unit | produced by | relations |
|---|---|---|---|---|
| `fid` | readout fidelity | — | `singleshot.GE_Exp` | — |
| `g_center` / `e_center` | ground / excited IQ centers | IQ | `GE_Exp` | discrimination |
| `ge_s` | single-state blob Gaussian σ (width) | IQ | `GE_Exp` | ge_radius / ge_s = detection radius in σ units |
| `ge_radius` | state-assignment detection radius | IQ | confusion-matrix calc | ge_radius / ge_s = detection radius in σ units |
| `confusion_matrix` | readout confusion matrix | — | `GE_Exp.calc_confusion_matrix` | corrects single-shot results |

## Cross-experiment relations & physics constants

The relations no single experiment owns — use them to seed or sanity-check values:

- **Sideband reset:** `reset_f = r_f − q_f`.
- **Cavity ring-down:** decay time τ = 1/(2π·`rf_w`); a `post_delay = 5/(2π·rf_w)` (≈5τ) lets
  the cavity empty before the next pulse — used across reset / CKP / AC-Stark / MIST.
- **Full relaxation:** `relax_delay ≈ 5·t1` when the experiment needs the qubit reset to |g⟩.
- **Flux period:** `flx_period = 2·abs(flx_int − flx_half)`.
- **JPA pump:** `best_jpa_freq ≈ 2·r_f` (plus a few-hundred-MHz offset).
- **Bath-reset qubit-tone gain:** `≈ min(1, 10·pi_gain·pi_len·rf_w)`.
- **Resonator photon number:** `n̄ = ac_stark_coeff · gain²` (resonator drive gain),
  calibrated from the AC-Stark shift; consumed by MIST / T1-with-tone analysis.
- **Predictor (FluxoniumPredictor):** `q_f = predict_freq(flux, (0,1))`; reset transitions feed
  `predict_freq(transition)`; `flx_bias = calculate_bias(...)`, fed back via `update_bias`.
