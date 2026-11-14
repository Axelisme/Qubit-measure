**Readers:** framework-dev. Read one scenario's measured outcomes and provenance.

# gate_sequence

Provenance: {"dirty":false,"exclusive":true,"head":"3545520dd8d5e8784be642246605f263654ce72b","started_at_utc":"2026-11-13T20:06:43.327493+00:00","tree":"c9f484253fae2cc0db5a7d80dc1dd21bbe808d77"}

Measured pinned families: t4, adiabatic_map, flux_dop853, slow_envelope, dyson, single_tone_rwa, floquet_period, t6, single_tone_gate, shaped_sambe, gate_cache, period_table, common_frame, multitone_gate, multitone_rwa, floquet, dop853.

Manifest: `10eae0a55dbbdf33bfd1c7faf1ce95a7d76fda3b52b789930e800db6bdba325e`. Physical: `a7a545d90fcbf8b346290f4cbd9ccee0ada2df60fddd2fd0bd9c688075ae5e5f`. Oracle: `7edd424818483d5d0b6d6bb4819c1a40c06bb5a7147ffaf75c9e20a95d466e8b`.

Times describe this table only; differences between runs do not establish faster/slower families.

| Policy | Status | Final density error | Attribution | Whitelist | Companions | Refusal |
| --- | --- | ---: | --- | --- | --- | --- |
| t4 | not_candidate | None | target_and_companions | ('t4',) | () | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| adiabatic_map | not_candidate | None | target_and_companions | ('adiabatic_map', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| flux_dop853 | not_candidate | None | target_and_companions | ('flux_dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| slow_envelope | not_candidate | None | target_and_companions | ('slow_envelope', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dyson | guard_or_precondition | None | target_and_companions | ('dyson', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| single_tone_rwa | qualified | 0.001483888891689632 | target_and_companions | ('single_tone_rwa', 't4') | ('t4',) | — |
| floquet_period | qualified | 2.3127606022641362e-07 | target_and_companions | ('floquet_period', 't4') | ('t4',) | — |
| t6 | qualified | 6.078532632444235e-05 | target_and_companions | ('t6', 't4') | ('t4',) | — |
| single_tone_gate | not_candidate | None | target_and_companions | ('single_tone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| shaped_sambe | not_candidate | None | target_and_companions | ('shaped_sambe', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| gate_cache | not_candidate | None | target_and_companions | ('gate_cache', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| period_table | not_candidate | None | target_and_companions | ('period_table', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| common_frame | not_candidate | None | target_and_companions | ('common_frame', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_gate | not_candidate | None | target_and_companions | ('multitone_gate', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| multitone_rwa | not_candidate | None | target_and_companions | ('multitone_rwa', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| floquet | not_candidate | None | target_and_companions | ('floquet', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| dop853 | not_candidate | None | target_and_companions | ('dop853', 't4') | ('t4',) | no valid kernel for segment 0; domain plateau: points [1], segments [0] |
| default | qualified | 0.001483888891689632 | default | None | () | — |
| fixed_default_group | qualified | 0.001483888891689632 | fixed_family_group | ('single_tone_rwa', 't4') | () | — |

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
| single_tone_rwa | first_evolve | 0.09810280400051852 | 0.09777109699999986 | 5 | — |
| single_tone_rwa | warm_clear_history | 0.09840866000013193 | 0.09806525499999985 | 5 | — |
| single_tone_rwa | warm_retain_history | 0.09818783399896347 | 0.09787789299999972 | 5 | — |
| floquet_period | first_evolve | 0.1422419760019693 | 0.14173815300000037 | 5 | — |
| floquet_period | warm_clear_history | 0.1422836109995842 | 0.1417449030000002 | 5 | — |
| floquet_period | warm_retain_history | 0.1421029390003241 | 0.14159597000000002 | 5 | — |
| t6 | first_evolve | 1.1390023970016045 | 1.136272292000001 | 5 | — |
| t6 | warm_clear_history | 1.1388938870004495 | 1.1361022619999979 | 5 | — |
| t6 | warm_retain_history | 1.1396446709986776 | 1.136563487 | 5 | — |
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
| default | first_evolve | 0.13906354400023702 | 0.13858975400000162 | 5 | — |
| default | warm_clear_history | 0.14209286800178234 | 0.1416175939999995 | 5 | — |
| default | warm_retain_history | 0.14038042899846914 | 0.1399495540000011 | 5 | — |
| fixed_default_group | first_evolve | 0.09797469499972067 | 0.09774496300000024 | 5 | — |
| fixed_default_group | warm_clear_history | 0.10118167199834716 | 0.10084784900000088 | 5 | — |
| fixed_default_group | warm_retain_history | 0.09917869499986409 | 0.09875343299999884 | 5 | — |
