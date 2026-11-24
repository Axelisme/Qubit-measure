**Readers:** framework-dev, agent. Use this ranking to find the slowest surveyed simulator scenes and the stage that costs the most.

# Simulator speed ranking

This is a historical ranking across commits. Blocks 1, 1b and 2 ran on different revisions and inputs. A difference between rows is not a speedup. Use the paired results in the last column to compare versions of the same scene. Follow the [maintenance rule](INDEX.md#收錄規則) before using a timing to choose work.

Core adds compile, evolve, readout and materialize time. The largest stage is the largest of those four totals. Cells are the completed Result points, not shots or routing rows. Families are listed by routing-row count, not by time; `—` means none recorded. All survey Runs used offline FTF on CPU, one worker, `time_scale=0`, with no repeat averaging. `Cold` means the first dispatched Run in its process; it does not isolate the JIT cost. Block 1b started a fresh process per timing and includes JIT cold cost where a signature compiled. A timeout has no Result denominator. Even where telemetry gives a partial core lower bound, do not divide by the planned grid.

The raw work files are not in the repository. The [tracked manifest](../../tools/data/experiment_wall_time/case-inputs.json) includes the surveyed Form overrides from blocks 1b and 2. The wall-time tool reads it with the adjacent fixture. The combined inputs do not have the same hash as any original batch. The GE classifiers, Local Z pair and shared CZ arb are for path timing only. The block 1 local mux scans used 101 × 11 Result cells. Block 1b audited an 11 × 11 Form grid without rerunning those scenes; the tracked manifest does not apply that audit's overrides.

## Survey inputs

The tables use SHA-256 prefixes. All three blocks used FTF settings `127360cb65e8117d71125328619667f04e3a0aea3d8c1adf12693bf74caa83d4` and board profile `f3e142655fc0056922488fd7748f79175a348f60eb9cb66048b3cec64fac5834`.

| Block and commit | Manifest SHA-256 | Input notes |
| --- | --- | --- |
| 1 · `593426c2fa384c42edc6197b07f6e2d78b764b48` | `931fa4f599f4db8dc82ac9363cb1068ac6f94ce9286e6e1baec34545bce57a1f` | `ftf-demo/initial`, no GE classifiers. Only the first dispatched Run was cold. |
| 1b · `381ce1607116064348558e5645382553f5cbe969` | `195f8ec5b12b6833a123123dcf6a709cb9569458e29b0ed2fc8fad2d8632095e` | Mux and onetone. |
| 1b · `381ce160` | `e7fc7a257da7ffdfb05d2ce748ef5a612a7a9c748a8b10766d03f1e8b476d779` | Twotone. |
| 1b · `381ce160` | `f1554d2ceba55596bfa20b37fbe88ef25e8bc2f87d538fc17afc265d41e9afbf` | Singleshot used a synthetic R1 GE circle classifier. |
| 1b · `381ce160` | `0d9bbfc93fa63535d3137e2c7b91fe90ba80412d76cea8be41b1d3bde9e8dbd9` | STIRAP Form audit, no Run. |
| 2 · `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a` | `c2b6b9e0522e50655a207a562a024ebfe9b87f88d37ead1490d46922d09344bc` | Two-qubit. The final input used simulated mux GE classifiers, a 0°/0° Local Z pair and 0.78 µs CZ pulses. Only CZ optimize used both lines with the shared `cz_survey` arb, SHA-256 `00444b3e34e6a91d32cf0e3b559059bc49b674e965e88770d6b7ad100de290c7`. This table takes one final Run per scene, not the earlier trials with different fixtures. |

## Completed scans

| Scene | Core s/cell | Largest stage | Families | Block · commit / manifest | Scope | Status | Task or attempt |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `chain/len_rabi` | 13.6952 | readout | `t4`, `floquet_period` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [ADC readout](#adc-readout) 106.060 → 93.163 core s; [Projective cell phase reuse](#projective-cell-phase-reuse) fewer source solves, no established core speedup |
| `two_qubit/conditional_spectroscopy_heralded` | 8.2706 | evolve | `t4`, `shaped_sambe`, `single_tone_rwa`, `floquet_period`, `dop853` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 11 cells, 1 Run, cold | `ok` | — |
| `twotone/ro_optimize/chain_power` | 7.0092 | materialize | `t4` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [ADC flat jump](#adc-flat-jump) 67.333 → 54.952 core s |
| `twotone/ro_optimize/chain_freq` | 6.9663 | materialize | `t4` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [ADC flat jump](#adc-flat-jump) 67.716 → 55.282 core s |
| `chain/amp_rabi` | 6.6886 | evolve | `t4`, `shaped_sambe` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, cold | `ok` | [ADC readout](#adc-readout) 67.767 → 66.059 core s; [Banded Sambe](#banded-sambe) 6.294 → 8.912 core s/cell |
| `twotone/ro_optimize/power` | 4.7089 | evolve | `t4`, `shaped_sambe`, `single_tone_gate` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [Banded Sambe](#banded-sambe) 3.903 → 4.893 core s/cell |
| `two_qubit/rb` | 4.2249 | evolve | `t4`, `shaped_sambe`, `single_tone_rwa`, `dop853` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 4 cells, 1 Run, cold | `ok` | [Same-carrier fold](#same-carrier-fold) 16.401 → 0.892 process CPU s |
| `two_qubit/cz_amplitude` | 2.3373 | evolve | `dop853`, `shaped_sambe`, `t4`, `single_tone_rwa` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 11 cells, 1 Run, cold | `ok` | — |
| `twotone/reset/check` | 1.7425 | evolve | `t4`, `shaped_sambe`, `single_tone_rwa`, `floquet_period` | 1b · `381ce160` / `e7fc7a257da7` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `mux/t2echo` | 1.5454 | evolve | `shaped_sambe`, `t4`, `dop853`, `multitone_rwa` | 1b · `381ce160` / `195f8ec5b12b` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `mux/len_rabi` | 1.1343 | evolve | `dop853`, `shaped_sambe`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [Banded Sambe](#banded-sambe) 1.091 → 1.099 core s/cell |
| `two_qubit/local_z` | 1.0932 | materialize | `t4`, `shaped_sambe`, `single_tone_rwa` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 5 cells, 1 Run, cold | `ok` | — |
| `mux/t1` | 1.0305 | evolve | `dop853`, `shaped_sambe`, `t4` | 1b · `381ce160` / `195f8ec5b12b` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `mux/t2ramsey` | 1.0016 | evolve | `shaped_sambe`, `t4`, `dop853`, `multitone_rwa` | 1b · `381ce160` / `195f8ec5b12b` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `two_qubit/conditional_spectroscopy` | 0.8188 | materialize | `t4`, `shaped_sambe`, `single_tone_gate`, `single_tone_rwa`, `dop853` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 11 cells, 1 Run, cold | `ok` | — |
| `mux/amp_rabi` | 0.7867 | evolve | `shaped_sambe`, `t4`, `multitone_rwa`, `dop853` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [Banded Sambe](#banded-sambe) 2.348 → 2.339 core s/cell |
| `twotone/ro_optimize/length` | 0.6469 | materialize | `shaped_sambe`, `t4` | 1b · `381ce160` / `e7fc7a257da7` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 1.576 → 1.713 core s (+8.7%); 0.987 → 1.037 materialize s |
| `singleshot/mist/power` | 0.5994 | materialize | `t4`, `shaped_sambe` | 1b · `381ce160` / `f1554d2ceba5` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 2.953 → 0.544 core s (-81.6%); 0.076 → 0.046 materialize s |
| `singleshot/mist/freq` | 0.5967 | materialize | `t4`, `shaped_sambe` | 1b · `381ce160` / `f1554d2ceba5` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 7.920 → 4.540 core s (-42.7%); 0.089 → 0.052 materialize s |
| `singleshot/reset_check` | 0.5426 | materialize | `t4`, `shaped_sambe`, `single_tone_rwa`, `floquet_period` | 1b · `381ce160` / `f1554d2ceba5` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 15.751 → 16.501 core s (+4.8%); 0.193 → 0.147 materialize s |
| `twotone/ro_optimize/freq` | 0.5383 | materialize | `shaped_sambe`, `t4` | 1b · `381ce160` / `e7fc7a257da7` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 0.523 → 0.470 core s (-10.1%); 0.398 → 0.334 materialize s |
| `singleshot/t1` | 0.5230 | materialize | `t4`, `shaped_sambe` | 1b · `381ce160` / `f1554d2ceba5` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 0.681 → 0.983 core s (+44.3%); 0.380 → 0.331 materialize s |
| `singleshot/amp_rabi` | 0.4965 | materialize | `shaped_sambe`, `t4` | 1b · `381ce160` / `f1554d2ceba5` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 0.252 → 0.456 core s (+80.7%); 0.096 → 0.197 materialize s |
| `twotone/rabi/amp_rabi` | 0.4941 | materialize | `shaped_sambe`, `t4` | 1b · `381ce160` / `e7fc7a257da7` | 11 cells, 1 Run, cold/JIT | `ok` | [Materialize follow-up](#materialize-follow-up) 0.321 → 0.265 core s (-17.4%); 0.069 → 0.057 materialize s |
| `two_qubit/idle_zz` | 0.4682 | materialize | `t4`, `shaped_sambe` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 11 cells, 1 Run, cold | `ok` | — |
| `two_qubit/chevron` | 0.4099 | evolve | `t4`, `multitone_gate`, `single_tone_gate`, `single_tone_rwa`, `shaped_sambe`, `dop853` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 121 cells, 1 Run, cold | `ok` | [Same-carrier fold](#same-carrier-fold) 143.688 → 143.713 process CPU s |
| `two_qubit/conditional_phase` | 0.3301 | materialize | `t4`, `single_tone_gate`, `single_tone_rwa`, `shaped_sambe` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 33 cells, 1 Run, cold | `ok` | — |
| `singleshot/mist/power_freq` | 0.2748 | readout | `t4`, `shaped_sambe` | 1b · `381ce160` / `f1554d2ceba5` | 121 cells, 1 Run, cold/JIT | `ok` | — |
| `twotone/ac_stark_ramsey` | 0.2562 | readout | `t4`, `shaped_sambe` | 1b · `381ce160` / `e7fc7a257da7` | 121 cells, 1 Run, cold/JIT | `ok` | — |
| `twotone/reset/flux/length` | 0.1994 | evolve | `t4`, `shaped_sambe` | 1 · `593426c2` / `931fa4f599f4` | 1 cells, 1 Run, warm | `ok` | — |
| `twotone/reset/flux/gain` | 0.1978 | evolve | `t4`, `shaped_sambe` | 1 · `593426c2` / `931fa4f599f4` | 1 cells, 1 Run, warm | `ok` | — |
| `twotone/rabi/len_rabi` | 0.1241 | materialize | `shaped_sambe`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | [Banded Sambe](#banded-sambe) 0.101 → 0.123 core s/cell |
| `twotone/ckp` | 0.0815 | readout | `t4`, `floquet_period`, `single_tone_gate` | 1 · `593426c2` / `931fa4f599f4` | 121 cells, 1 Run, warm | `ok` | — |
| `twotone/flux_dep` | 0.0775 | evolve | `t4`, `single_tone_rwa`, `dyson`, `floquet_period` | 1 · `593426c2` / `931fa4f599f4` | 121 cells, 1 Run, warm | `ok` | — |
| `twotone/ro_optimize/freq_gain` | 0.0766 | materialize | `t4`, `single_tone_gate`, `shaped_sambe` | 1b · `381ce160` / `e7fc7a257da7` | 121 cells, 1 Run, cold/JIT | `ok` | — |
| `singleshot/len_rabi` | 0.0623 | evolve | `shaped_sambe`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | — |
| `twotone/ac_stark` | 0.0471 | evolve | `floquet_period`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 121 cells, 1 Run, warm | `ok` | — |
| `twotone/cpmg` | 0.0434 | evolve | `t4`, `single_tone_gate`, `shaped_sambe` | 1b · `381ce160` / `e7fc7a257da7` | 55 cells, 1 Run, cold/JIT | `ok` | — |
| `mux/onetone_flux_local` | 0.0415 | readout | — | 1 · `593426c2` / `931fa4f599f4` | 1111 cells, 1 Run, warm | `ok` | — |
| `twotone/dispersive` | 0.0310 | compile | `shaped_sambe`, `t4` | 1b · `381ce160` / `e7fc7a257da7` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `twotone/power_dep` | 0.0262 | evolve | `t4`, `floquet_period`, `single_tone_rwa`, `dyson` | 1 · `593426c2` / `931fa4f599f4` | 121 cells, 1 Run, warm | `ok` | — |
| `mux/onetone_flux_global` | 0.0234 | readout | — | 1 · `593426c2` / `931fa4f599f4` | 121 cells, 1 Run, warm | `ok` | — |
| `twotone/reset/single_tone/freq` | 0.0204 | evolve | `t4`, `shaped_sambe`, `single_tone_rwa` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | — |
| `mux/onetone` | 0.0202 | readout | — | 1b · `381ce160` / `195f8ec5b12b` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `mux/twotone_flux_local` | 0.0178 | evolve | `multitone_rwa`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 1111 cells, 1 Run, warm | `ok` | — |
| `mux/twotone_spectrum` | 0.0158 | evolve | `multitone_rwa`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | — |
| `mux/twotone_flux_global` | 0.0119 | evolve | `multitone_rwa`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 121 cells, 1 Run, warm | `ok` | — |
| `twotone/freq` | 0.0089 | evolve | `t4`, `single_tone_rwa`, `floquet_period` | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | — |
| `onetone/freq` | 0.0060 | evolve | — | 1b · `381ce160` / `195f8ec5b12b` | 11 cells, 1 Run, cold/JIT | `ok` | — |
| `onetone/flux_dep` | 0.0037 | readout | — | 1b · `381ce160` / `195f8ec5b12b` | 121 cells, 1 Run, cold/JIT | `ok` | — |
| `onetone/range_find` | 0.0037 | readout | — | 1 · `593426c2` / `931fa4f599f4` | 11 cells, 1 Run, warm | `ok` | — |
| `onetone/power_dep` | 0.0032 | readout | — | 1b · `381ce160` / `195f8ec5b12b` | 121 cells, 1 Run, cold/JIT | `ok` | — |

## Shot-only runs

These scenes report core time for the whole Run. They have no scan denominator and do not enter the ranking.

| Scene | Core s total | Largest stage | Families | Block · commit / manifest | Scope | Status | Task or attempt |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `twotone/rb` | 41.947 | evolve | `single_tone_gate`, `t4`, `shaped_sambe` | 1b · `381ce160` / `e7fc7a257da7` | 1 Run, cold/JIT, depths `[0,1,2,4,8,16,32,64,128]`, 1000 reps × 10 rounds | `ok` | — |
| `mux/singleshot_ge` | 25.942 | evolve | `dop853`, `shaped_sambe`, `t4` | 1b · `381ce160` / `195f8ec5b12b` | 1 Run, cold/JIT, 100,000 shots | `ok` | — |
| `two_qubit/joint_readout` | 6.378 | materialize | `t4`, `shaped_sambe`, `dop853` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 1 Run, cold | `ok` | — |
| `singleshot/ge` | 5.526 | materialize | `shaped_sambe`, `t4` | 1b · `381ce160` / `f1554d2ceba5` | 1 Run, cold/JIT, 100,000 shots | `ok` | — |
| `twotone/allxy` | 5.465 | materialize | `t4`, `shaped_sambe` | 1b · `381ce160` / `e7fc7a257da7` | 1 Run, cold/JIT, 1000 reps × 10 rounds | `ok` | — |
| `singleshot/check` | 5.413 | materialize | `t4`, `floquet_period`, `shaped_sambe` | 1b · `381ce160` / `f1554d2ceba5` | 1 Run, cold/JIT, 5000 shots | `ok` | — |
| `singleshot/qutrit_population` | 5.338 | materialize | `t4`, `floquet_period`, `shaped_sambe` | 1b · `381ce160` / `f1554d2ceba5` | 1 Run, cold/JIT, 4096 shots | `ok` | — |

## Unranked scenes

A stage or family on a timeout row describes only the recorded work. A dash in the score column is not zero. The survey did not select adapters outside these tables.

| Scene | Core s/cell | Largest recorded stage | Recorded families | Block · commit / manifest | Scope | Status | Task or attempt |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| `chain/t1` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s | [ADC readout](#adc-readout) 122.660 → 107.872 core s |
| `chain/t2echo` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s | [ADC readout](#adc-readout) 192.778 → 158.262 core s |
| `chain/t2ramsey` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s | [ADC readout](#adc-readout) 135.638 → 108.336 core s |
| `singleshot/ac_stark` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | uncovered: Run failed, GE classifier missing | — |
| `singleshot/mist/pre_freq` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | uncovered: Run failed, GE classifier missing | — |
| `singleshot/t1_tone` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | uncovered: Run failed, GE classifier missing | — |
| `singleshot/t1_tone_sweep_freq` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | uncovered: Run failed, GE classifier missing | — |
| `singleshot/t1_tone_sweep_gain` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | uncovered: Run failed, GE classifier missing | — |
| `two_qubit/conditional_phase_heralded` | — | evolve | `shaped_sambe`, `t4`, `single_tone_rwa` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 1 attempt, cold, no Result cells | timeout: ≥180 wall s; ≥116.970 core s | — |
| `two_qubit/constructive_phase` | — | evolve | `t4`, `single_tone_gate`, `multitone_gate`, `single_tone_rwa`, `shaped_sambe`, `dop853`, `multitone_rwa`, `dyson` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 1 attempt, cold, no Result cells | timeout: ≥180 wall s; ≥183.776 core s | — |
| `two_qubit/cz_optimize` | — | — | — | 2 · `addd6c8fb` / `c2b6b9e0522e` | 1 attempt, cold, no Result cells | uncovered: Run failed, generator 2 buffer exceeded 8192 samples | [Same-carrier fold](#same-carrier-fold) 3.108 → 0.140 process CPU s |
| `two_qubit/rb_heralded` | — | evolve | `t4`, `shaped_sambe`, `single_tone_rwa`, `dop853` | 2 · `addd6c8fb` / `c2b6b9e0522e` | 1 attempt, cold, no Result cells | timeout: ≥180 wall s; ≥119.475 core s | — |
| `twotone/chain` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s | [ADC readout](#adc-readout): both sides timed out |
| `twotone/raman_gf` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing Raman waveform data and carrier | — |
| `twotone/reset/bath/freq_gain` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing cavity frequency start/stop | — |
| `twotone/reset/bath/length` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing cavity tone frequency/gain | — |
| `twotone/reset/bath/phase` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing cavity tone frequency/gain | — |
| `twotone/reset/dual_tone/freq` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing both pulse gains and frequency windows | — |
| `twotone/reset/dual_tone/length` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing both pulse frequencies and gains | — |
| `twotone/reset/dual_tone/power` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing both pulse frequencies | — |
| `twotone/reset/single_tone/length` | — | — | — | 1b · `381ce160` / `e7fc7a257da7` | not dispatched | uncovered: Form missing pulse frequency | — |
| `twotone/ro_optimize/auto` | — | evolve | `shaped_sambe`, `t4` | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s; ≥77.777 core s | — |
| `twotone/ro_optimize/chain_length` | — | materialize | `t4` | 1b · `381ce160` / `e7fc7a257da7` | 1 recorded cold/JIT retry, no Result cells | timeout: ≥180 wall s; ≥161.826 core s | — |
| `twotone/ro_optimize/rpi_freq` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s | — |
| `twotone/ro_optimize/rpi_length` | — | readout | `t4`, `shaped_sambe`, `single_tone_gate` | 1 · `593426c2` / `931fa4f599f4` | 1 attempt, warm, no Result cells | timeout: ≥180 wall s; ≥170.842 core s | — |
| `twotone/ro_optimize/rpi_power` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | 1 interrupted attempt, warm batch, no Result cells | uncovered: interrupted, no terminal row | — |
| `twotone/stirap/amplitude` | — | — | — | 1b · `381ce160` / `0d9bbfc93fa6` | not dispatched | uncovered: Form missing pump/stokes carriers and gain windows | — |
| `twotone/stirap/delay` | — | — | — | 1b · `381ce160` / `0d9bbfc93fa6` | not dispatched | uncovered: Form missing pump/stokes carriers and gains | — |
| `twotone/stirap/detuning` | — | — | — | 1b · `381ce160` / `0d9bbfc93fa6` | not dispatched | uncovered: Form missing pump/stokes carriers, gains and detuning windows | — |
| `twotone/t1` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |
| `twotone/t1_cov` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |
| `twotone/t2echo` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |
| `twotone/t2ramsey` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |
| `twotone/zigzag` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |
| `twotone/zigzag_scan/freq` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |
| `twotone/zigzag_scan/gain` | — | — | — | 1 · `593426c2` / `931fa4f599f4` | not dispatched | uncovered: batch stopped before dispatch | — |

## Paired measurements

The pairs show a change tested on fixed inputs. They do not overwrite the historical survey column. The before/after commits are measurement revisions, not necessarily the landing merge commits.

### ADC readout

[ADC cavity-product integrals](2026-11-24-adc-cavity-product-integrals.md#實測) compares `addd6c8fb` → `2a571d682` on manifest `931fa4f599f4`. One cold-cache, 11-cell Run per side and scene, same CPU slot, offline fixture, FTF seed 123, one worker. Evolve was the largest stage in all completed pairs. `twotone/chain` reached the 180 s Run deadline on both sides with no complete core or Result.

### ADC flat jump

[ADC flat jump baselines](2026-11-24-adc-flat-jump-baselines.md#實測) compares `addd6c8fb` → `5e43af2c7` on manifest `931fa4f599f4`. One cold-cache, 11-cell Run per side and scene, same CPU slot, offline fixture, FTF seed 123, one worker. Materialize remained the largest stage in both scenes.

### Banded Sambe

[The rejected banded Sambe candidate](2026-11-24-banded-shaped-sambe.md#實測) compares `addd6c8fb` → `0f5d83887` on manifest `931fa4f599f4`. Each side started a batch from an empty cache and ran each scene once, 11 cells, FTF seed 123, one worker and one native thread. The mux and twotone duration bounds used in this pair are recorded in the attempt.

### Same-carrier fold

[Same-carrier folding](2026-11-23-same-carrier-merge.md#實測) compares `2b8f628c3` → `0d851985`. The record identifies workloads, not a manifest hash. Every workload ran once per side after one warmup, on CPU with one worker. Plain RB used depths `[0,4,8]`, sequence seed 701, sim seed 123, two reps and one round. CZ optimize used only E0, with 24 retained states and raw gain. Chevron used 2 × 2 cells, gg herald, 100 reps and one round. The CPU totals are for these workloads only, not the survey grids.

### Projective cell phase reuse

[The reduced pair](2026-11-24-projective-cell-phase-reuse.md#實測) compares `f7a88acef` → `c22947414` on manifest `589a368693c2`. Each side ran one cold-cache `chain/len_rabi` Run with 11 points, 100 reps, 100 rounds, seed 123 and one CPU worker in the same exclusive slot. The first preparation solved 3,229 → 2,967 source rows; the maps and Result matched. Full Run core was 96.324 → 96.032 s. With one observation per side, this does not establish a whole-Run speedup. The historical survey used a different manifest and a warm Run.

### Materialize follow-up

The warm follow-up compares `381ce160` → `387b988e`, manifest `3b73a371d78935593e0376931af12dcdb448b42e3fada394ddafc190a606cd0a`. Both sides warmed up `singleshot/amp_rabi` then ran each of the eight named scenes once, 11 Result cells, seed 123 and one worker. Only four scenes showed a core reduction over 5% in these single pairs. The results did not preserve shot moments for a parity check. [ADC flat jump baselines](2026-11-24-adc-flat-jump-baselines.md) records the optimization and its separate parity and speed comparisons.

## Pulse probes are separate

The survey also timed one `Engine.inspect_dispatch` plus one `Engine.evolve` per pulse cell. Those CPU seconds are not Run core seconds, so they do not enter the ranking. For same-carrier `same-flat_top-100ns-weak` and `drag-cosine-100ns-weak`, block 2 recorded 11.044 and 8.257 CPU s at `addd6c8fb`, a single sample per cell in a cold-started sweep. Block 1b recorded 0.025545 and 0.060731 CPU s at `381ce160` after one warmup per cell. Both used FTF input SHA-256 `127360cb65e8117d71125328619667f04e3a0aea3d8c1adf12693bf74caa83d4`, a 24-state model, two +5.800 GHz lines, 100 ns and 0.1× the estimated π gain. The process warmth and revision differ, so the pairs do not measure a kernel speedup. See the [same-carrier shaped Sambe](2026-11-24-same-carrier-shaped-sambe.md) and [long dual-carrier waveform](2026-11-24-long-dual-carrier-builtins.md) attempts for the routing follow-ups.
