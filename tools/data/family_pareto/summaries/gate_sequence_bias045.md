**Readers:** framework-dev. Read one scenario's measured outcomes and provenance.

# gate_sequence_bias045

Provenance: {"dirty":false,"exclusive":true,"head":"3545520dd8d5e8784be642246605f263654ce72b","started_at_utc":"2026-11-13T22:42:02.303631+00:00","tree":"c9f484253fae2cc0db5a7d80dc1dd21bbe808d77"}

Measured pinned families: t4, adiabatic_map, flux_dop853, slow_envelope, dyson, single_tone_rwa, floquet_period, t6, single_tone_gate, shaped_sambe, gate_cache, period_table, common_frame, multitone_gate, multitone_rwa, floquet, dop853.

Manifest: `b106578dbc205d08ef033bcb0fd95369b999d6f0160dbb8ef1969b6eb7d748e3`. Physical: `032d8726c090cba8ee3d91ebdbebd268bc38b6a65d991e6acfb491ad4a58df04`. Oracle: `bf7254cd0b46aebced2c51557d9f65b359f755da7d4ecac1e1ff769684982a33`.

Times describe this table only; differences between runs do not establish faster/slower families.

| Policy | Status | Final density error | Attribution | Whitelist | Companions | Refusal |
| --- | --- | ---: | --- | --- | --- | --- |
| t4 | not_candidate | None | target_and_companions | ('t4',) | () | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | not_candidate | None | target_and_companions | ('adiabatic_map', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | not_candidate | None | target_and_companions | ('flux_dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | not_candidate | None | target_and_companions | ('slow_envelope', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | guard_or_precondition | None | target_and_companions | ('dyson', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_rwa | over_limit | 0.003459452967093288 | target_and_companions | ('single_tone_rwa', 't4') | ('t4',) | — |
| floquet_period | qualified | 3.9948960646981396e-06 | target_and_companions | ('floquet_period', 't4') | ('t4',) | — |
| t6 | qualified | 0.00012771142295613458 | target_and_companions | ('t6', 't4') | ('t4',) | — |
| single_tone_gate | not_candidate | None | target_and_companions | ('single_tone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | not_candidate | None | target_and_companions | ('shaped_sambe', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | not_candidate | None | target_and_companions | ('gate_cache', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | not_candidate | None | target_and_companions | ('period_table', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | not_candidate | None | target_and_companions | ('common_frame', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | not_candidate | None | target_and_companions | ('multitone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | not_candidate | None | target_and_companions | ('multitone_rwa', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | not_candidate | None | target_and_companions | ('floquet', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | not_candidate | None | target_and_companions | ('dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| default | over_limit | 0.003459452967093288 | default | None | () | — |
| fixed_default_group | over_limit | 0.003459452967093288 | fixed_family_group | ('single_tone_rwa', 't4') | () | — |

| Policy | Mode | Wall median s | Process CPU median s | Samples | Unmeasured reason |
| --- | --- | ---: | ---: | ---: | --- |
| t4 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| t4 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| t4 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_rwa | first_evolve | 0.09949314099503681 | 0.09912303500000008 | 5 | — |
| single_tone_rwa | warm_clear_history | 0.09779686800175114 | 0.09742854699999981 | 5 | — |
| single_tone_rwa | warm_retain_history | 0.10068943799706176 | 0.100389909 | 5 | — |
| floquet_period | first_evolve | 0.8695413740060758 | 0.8663471199999995 | 5 | — |
| floquet_period | warm_clear_history | 0.8768502169987187 | 0.8733713049999992 | 5 | — |
| floquet_period | warm_retain_history | 0.8727848449998419 | 0.8695257759999997 | 5 | — |
| t6 | first_evolve | 1.240719387998979 | 1.2366647630000003 | 5 | — |
| t6 | warm_clear_history | 1.2570499579960597 | 1.2505278290000064 | 5 | — |
| t6 | warm_retain_history | 1.2482568549967255 | 1.2429699119999995 | 5 | — |
| single_tone_gate | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_gate | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_gate | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | first_evolve | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | warm_clear_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | warm_retain_history | None | None | 0 | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| default | first_evolve | 0.1394633710006019 | 0.1387223930000019 | 5 | — |
| default | warm_clear_history | 0.14166283400118118 | 0.1408356640000008 | 5 | — |
| default | warm_retain_history | 0.14212413799396018 | 0.14143261099999904 | 5 | — |
| fixed_default_group | first_evolve | 0.09853149000264239 | 0.0980884599999996 | 5 | — |
| fixed_default_group | warm_clear_history | 0.0993226999999024 | 0.09890189800000115 | 5 | — |
| fixed_default_group | warm_retain_history | 0.09888543000124628 | 0.09848624100000336 | 5 | — |
